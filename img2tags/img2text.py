#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import deque  # not thread-safe
from collections.abc import Iterator
from pathlib import Path

from img2tags.common import track

if "--online" not in sys.argv:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
from PIL import Image, ImageFile
from transformers import BatchFeature, BitsAndBytesConfig, Blip2ForConditionalGeneration, Blip2Processor

SUPPORTED_DTYPES: list[str] = [
    "bf16",
    "nf4",
]
ImageFile.LOAD_TRUNCATED_IMAGES = True
PATH_STDIN: Path = Path("/dev/stdin")


class Captioner:
    processor: Blip2Processor

    def __init__(
        self,
        *,
        model_name: str,
        dtype: str,
        path_query: Path | None,
        path_config: Path | None,
        batch_size: int,
        disable_dynamic_config: bool,
    ):
        Image.MAX_IMAGE_PIXELS = 3_000_000_000
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Blip2ForConditionalGeneration
        self.batch_size = batch_size
        self.disable_dynamic_config = disable_dynamic_config
        if dtype == "nf4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            self.model = Blip2ForConditionalGeneration.from_pretrained(  # type: ignore
                model_name,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            self.model = Blip2ForConditionalGeneration.from_pretrained(  # type: ignore
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        self.processor = Blip2Processor.from_pretrained(model_name)  # type: ignore

        if path_config is None:
            self.config = {
                "max_length": 15,
                "min_length": 0,
                "do_sample": False,
                "num_beams": 4,
                "repetition_penalty": 1.5,
            }
        else:
            with path_config.open() as inf:
                self.config = json.load(inf)

        if path_query is None:
            self.qs = None
            self.qs_encodedd = None
        else:
            self.update_qs(
                path_query=path_query,
            )
        self.image_queue = deque()

    def update_qs(
        self,
        *,
        path_query: Path,
    ) -> None:
        qs: list[str] = []
        with path_query.open() as inf:
            lines = inf.readlines()
            if len(lines) == 0 or (len(lines) == 1 and len(lines[0]) == 0):
                self.qs = None
                self.qs_encodedd = None
                return
            for line in lines:
                q = line.strip()
                if len(q) == 0 or line.startswith("#"):
                    continue
                qs.append(q)
        self.qs = qs
        self.qs_encoded = self.processor(
            text=self.qs * self.batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=75,
        )

    def run(
        self,
        *,
        unit: int | None = None,
    ) -> Iterator[dict]:
        myunit: int = self.batch_size if unit is None else unit
        while len(self.image_queue) >= myunit:
            j: int = 0
            names: list[str] = []
            images: list = []
            while j < len(self.image_queue):
                target, input_ienc = self.image_queue[j]
                j += 1

                if isinstance(input_ienc, str):
                    yield {
                        "input": str(target),
                        "error": input_ienc,
                    }
                    self.image_queue.popleft()
                    continue
                names.append(str(target))
                images.append(input_ienc["pixel_values"][0])
                if len(images) == myunit:
                    for _ in range(myunit):
                        self.image_queue.popleft()
                    break
            else:
                return

            original_tensor = torch.stack(images, dim=0)

            ienc = BatchFeature()
            if self.qs:
                ienc["pixel_values"] = original_tensor.repeat(len(self.qs), 1, 1, 1)
                if myunit == self.batch_size:
                    ienc.update(self.qs_encoded)
                else:
                    __qs_encoded = self.processor(
                        text=self.qs * self.batch_size,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=75,
                    )
                    ienc.update(__qs_encoded)
            else:
                ienc["pixel_values"] = original_tensor

            ienc.to(
                self.device,
                dtype=torch.bfloat16,  # type: ignore
            )
            generated = self.model.generate(**ienc, **self.config)
            texts: list[str] = [
                v.replace("\n", " ").strip()
                for v in self.processor.batch_decode(
                    generated,
                    skip_special_tokens=True,
                )
            ]

            if self.qs is None:
                for name, text in zip(names, texts, strict=True):  # type: ignore
                    yield {"input": name, "result": [text]}
                return

            rets = [{"input": name, "result": []} for name in names]

            for i, v in enumerate(texts):
                rets[i % myunit]["result"].append(v)

            yield from rets

    def finish_queue(self) -> Iterator[dict]:
        yield from self.run(unit=len(self.image_queue))

    def add_task(
        self,
        *,
        target: Path,
    ):
        if not self.disable_dynamic_config:
            if target.suffix == ".txt":
                self.finish_queue()
                self.update_qs(
                    path_query=target,
                )
                return
            elif target.suffix == ".json":
                self.finish_queue()
                with target.open() as inf:
                    self.config = json.load(inf)
                return

        try:
            image = Image.open(str(target)).convert("RGB")
            image_encoded = self.processor(
                image,
                return_tensors="pt",
            )
            self.image_queue.append(
                (
                    target,
                    image_encoded,
                )
            )
        except Exception as e:
            self.image_queue.append(
                (
                    target,
                    str(e),
                )
            )


def iter_input(path_in: Path):
    if path_in.is_dir():
        files: list[Path] = [v for v in path_in.iterdir()]
        yield from track(files)
        return

    with_progress_bar: bool = bool(PATH_STDIN != path_in)
    with path_in.open() as inf:
        if with_progress_bar:
            itr = track(inf)
        else:
            itr = inf

        for line in itr:
            line = line[:-1]
            if len(line) == 0:
                continue
            if line.startswith("'") and line.endswith("' "):
                line = line[1:-2]
            file_url: Path = Path(line).expanduser()
            yield file_url


def operation(
    *,
    path_in: Path,
    path_out: Path,
    model_name: str,
    dtype: str,
    batch_size: int,
    path_query: Path,
    path_config: Path,
    disable_dynamic_config: bool,
) -> None:
    cpt: Captioner = Captioner(
        model_name=model_name,
        dtype=dtype,
        path_query=path_query,
        path_config=path_config,
        batch_size=batch_size,
        disable_dynamic_config=disable_dynamic_config,
    )

    with path_out.open("w") as outf:
        for file_url in iter_input(path_in):
            cpt.add_task(target=file_url)
            for d in cpt.run():
                outf.write(json.dumps(d, ensure_ascii=False))
                outf.write("\n")
                outf.flush()
        for d in cpt.finish_queue():
            outf.write(json.dumps(d, ensure_ascii=False))
            outf.write("\n")
            outf.flush()


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, default=PATH_STDIN, required=False)
    oparser.add_argument("--output", "-o", type=Path, default="/dev/stdout", required=False)
    oparser.add_argument(
        "--model",
    )
    oparser.add_argument(
        "--config",
        type=Path,
    )

    oparser.add_argument(
        "--dtype",
        choices=SUPPORTED_DTYPES,
        default="nf4",
    )
    oparser.add_argument(
        "--batch_size",
        "--bs",
        type=int,
        default=1,
    )
    oparser.add_argument(
        "--online",
        action="store_true",
    )
    oparser.add_argument(
        "--disable_dynamic_config",
        action="store_true",
    )
    oparser.add_argument(
        "--query",
        "-q",
        type=Path,
    )
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()

    model: str = "Salesforce/blip2-opt-6.7b-coco"
    if opts.query is None:
        model = "Salesforce/blip2-opt-6.7b"

    operation(
        path_in=opts.input,
        path_out=opts.output,
        model_name=model,
        dtype=opts.dtype,
        batch_size=opts.batch_size,
        path_query=opts.query,
        path_config=opts.config,
        disable_dynamic_config=opts.disable_dynamic_config,
    )


if __name__ == "__main__":
    main()
