#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Final

if "--offline" in sys.argv:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


import numpy as np
import onnxruntime as rt
import torch
import torch.utils.data
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFile

from img2tags.common import setup_logging, track
from img2tags.tagger import ImageTagger, ImageTaggerConfig, ResultConverter

setup_logging()
ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


class ImageLoadingPrepDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        image_size: int,
    ):
        self.images = image_paths
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            image = ImageTagger.preprocess(
                image=image,
                image_size=self.image_size,
            )
            tensor = torch.tensor(image)
        except Exception as e:
            logger.warning(f"Load error: {img_path}, error: {e}")
            return (None, img_path)
        return (tensor, img_path)


def collate_fn_remove_corrupted(batch):
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".gif",
    ".jpeg",
    ".webp",
    ".bmp",
    ".avif",
}

OUTPUT_EXTENSIONS = {
    "txt",
    "json",
    "jsonl",
}


def glob_images(path_target: Path, exp: str):
    for p in path_target.glob(exp):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


MODEL_FILE_NAME: Final[str] = "model.onnx"
CONFIG_FILE_NAME: Final[str] = "config.json"


MODELS_ONLY_BS1: Final[set[str]] = {
    "SmilingWolf/wd-v1-4-convnext-tagger-v2",
    "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
    "SmilingWolf/wd-v1-4-moat-tagger-v2",
}


def run(
    *,
    path_or_name_model: str,
    path_in: Path,
    path_out: Path | None,
    max_data_loader_n_workers: int,
    batch_size: int,
    ext: str,
    threshold_str: str | None,
    force_cpu: bool,
):
    if path_or_name_model in MODELS_ONLY_BS1:
        assert batch_size == 1, f"Only batch_size = 1 is supported for {path_or_name_model}"
    assert path_in.exists(), f"Not found: {path_in}"

    path_config: Path | None = None
    path_model = Path(path_or_name_model)
    if path_model.exists():
        path_config = path_model.joinpath("config.json")
        path_model = path_model.joinpath("model.onnx")
    else:
        items: list[str] = path_or_name_model.split("/")
        assert len(items) >= 2
        repo_id: str = "/".join(items[:2])
        if len(items) > 2:
            # eg: shirayu/img2tags/SmilingWolf__wd-v1-4-convnext-tagger-v2
            filename = "/".join(items[2:]) + f"/{MODEL_FILE_NAME}"

            path_config = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename="/".join(items[2:]) + f"/{CONFIG_FILE_NAME}",
                )
            )

        else:
            # eg: SmilingWolf/wd-v1-4-moat-tagger-v2
            filename = MODEL_FILE_NAME

        path_model = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )

    config: ImageTaggerConfig
    if path_config:
        with path_config.open() as inf:
            config = ImageTaggerConfig.model_validate_strings(inf.read())
    else:
        default_threthold = {
            "SmilingWolf/wd-v1-4-convnext-tagger-v2": 0.3685,
            "SmilingWolf/wd-v1-4-convnextv2-tagger-v2": 0.3710,
            "SmilingWolf/wd-v1-4-moat-tagger-v2": 0.3771,
            "SmilingWolf/wd-vit-tagger-v3": 0.2547,  # F1 = 0.4278
            "SmilingWolf/wd-swinv2-tagger-v3": 0.2521,  # F1 = 0.4411
            "SmilingWolf/wd-convnext-tagger-v3": 0.2570,  # F1 = 0.4282
            "SmilingWolf/wd-vit-large-tagger-v3": 0.2606,  # F1 = 0.4674
            "SmilingWolf/wd-eva02-large-tagger-v3": 0.5296,  # F1 = 0.4772
        }[path_or_name_model]
        config = ImageTaggerConfig(
            image_size=448,
            thresholds_for_json={
                0: default_threthold,
                4: default_threthold,
                9: 0.0,
            },
            thresholds_for_txt={
                0: default_threthold,
                4: default_threthold,
                9: 10000,
            },
        )

    providers: list[str] = ["CPUExecutionProvider"]
    if not force_cpu and "CUDAExecutionProvider" in rt.get_available_providers():
        providers = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    model: rt.InferenceSession = rt.InferenceSession(
        path_model,
        providers=providers,
    )

    image_paths: list[Path] = []
    if path_in.is_dir():
        image_paths = list(glob_images(path_in, "**/*"))
    else:
        with path_in.open() as inf:
            for line in inf:
                image_paths.append(Path(line[:-1]))

    rconv = ResultConverter(path_or_name_model=path_or_name_model)
    is_json: bool = ext.lower() == "json"
    is_jsonl: bool = ext.lower() == "jsonl"

    thresholds: dict[int, float]
    if threshold_str is None:
        if is_json or is_jsonl:
            thresholds = config.thresholds_for_json
        else:
            thresholds = config.thresholds_for_txt
    else:
        thresholds = json.loads(threshold_str)

    single_outf = None
    if is_jsonl:
        if path_out is None:
            single_outf = Path("/dev/stdout").open("w")
        else:
            single_outf = path_out.open("w")

    def run_batch(img_pairs):
        imgs = np.array([im for _, im in img_pairs])

        input_name = model.get_inputs()[0].name
        probs = model.run(None, {input_name: imgs})[0]

        for (image_path, _), prob in zip(img_pairs, probs, strict=True):  # type: ignore
            result = rconv(
                prob=prob,
                thresholds=thresholds,
            )

            if is_jsonl:
                assert single_outf is not None
                d = result.model_dump()
                d["input"] = str(image_path)
                single_outf.write(json.dumps(d, ensure_ascii=False))
                single_outf.write("\n")
                continue

            my_path_out: Path = image_path.parent
            if path_out is not None:
                rel: Path = image_path.parent.relative_to(path_in)
                my_path_out = path_out.joinpath(rel)
                my_path_out.mkdir(exist_ok=True, parents=True)

            with my_path_out.joinpath(image_path.stem + f".{ext}").open("w") as outf:
                if is_json:
                    outf.write(json.dumps(result.model_dump(), indent=4, ensure_ascii=False))
                else:
                    kvs: list[tuple[str, float]] = []
                    outs: list[str] = []
                    for c, tv in result.tags.items():
                        if c == 9:  # skip rating
                            continue
                        for t, v in tv.items():
                            kvs.append((t, v))
                    for t in sorted(kvs, key=lambda x: x[1], reverse=True):
                        outs.append(t[0])
                    outf.write(", ".join(outs))

                outf.write("\n")

    dataset = ImageLoadingPrepDataset(image_paths, image_size=config.image_size)
    data = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max_data_loader_n_workers,
        collate_fn=collate_fn_remove_corrupted,
        drop_last=False,
    )

    b_imgs = []
    for data_entry in track(data):
        for data in data_entry:
            if data is None:
                continue

            image, image_path = data
            if image is not None:
                image = image.detach().numpy()
            else:
                try:
                    image = Image.open(image_path)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image = ImageTagger.preprocess(
                        image=image,
                        image_size=config.image_size,
                    )
                except Exception as e:
                    logger.warning(f"Load error: {image_path}, error: {e}")
                    continue
            b_imgs.append((image_path, image))

            if len(b_imgs) >= batch_size:
                b_imgs = [(image_path, image) for image_path, image in b_imgs]  # Convert image_path to string
                run_batch(b_imgs)
                b_imgs.clear()

    if len(b_imgs) > 0:
        b_imgs = [(image_path, image) for image_path, image in b_imgs]  # Convert image_path to string
        run_batch(b_imgs)


def get_opts() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="SmilingWolf/wd-eva02-large-tagger-v3",
    )
    parser.add_argument(
        "--batch_size",
        "--batch",
        "--bs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="txt",
        choices=OUTPUT_EXTENSIONS,
    )
    parser.add_argument(
        "--thresholds",
        "--th",
        type=str,
    )
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
    )

    parser.add_argument("--cpu", action="store_true", help="Force to use CPU")
    parser.add_argument("--offline", action="store_true", help="Set offline mode")

    return parser


def get_version(pkg_name: str) -> str:
    from importlib.metadata import distributions

    for dist in distributions():
        if dist.metadata["Name"] == pkg_name:
            return dist.metadata["Version"]
    return "Unknown"


def main() -> None:
    parser = get_opts()
    opts = parser.parse_args()
    if not opts.version and not opts.input:
        parser.error("-i is required unless -v is specified")

    if opts.version:
        print(get_version("img2tags"))
        return

    run(
        path_or_name_model=opts.model,
        path_in=opts.input,
        path_out=opts.output,
        max_data_loader_n_workers=opts.max_data_loader_n_workers,
        batch_size=opts.batch_size,
        ext=opts.ext,
        threshold_str=opts.thresholds,
        force_cpu=opts.cpu,
    )


if __name__ == "__main__":
    main()
