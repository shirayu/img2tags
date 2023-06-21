#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path
from typing import Final, Optional

import numpy as np
import onnxruntime as rt
import torch
import torch.utils.data
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

from img2tags.tagger import ImageTagger, ImageTaggerConfig, ResultConverter

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

        image = Image.open(img_path).convert("RGB")
        image = ImageTagger.preprocess(
            image=image,
            image_size=self.image_size,
        )
        tensor = torch.tensor(image)
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
}


def glob_images(path_target: Path, exp: str):
    for p in path_target.glob(exp):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


MODEL_FILE_NAME: Final[str] = "model.onnx"
CONFIG_FILE_NAME: Final[str] = "config.json"


def run(
    *,
    path_or_name_model: str,
    path_in: Path,
    path_out: Optional[Path],
    max_data_loader_n_workers: int,
    batch_size: int,
    ext: str,
    threshold_str: Optional[str],
    force_cpu: bool,
):
    assert path_in.exists(), f"Not found: {path_in}"
    assert path_in.is_dir(), f"Not directory: {path_in}"

    path_config: Path
    path_model = Path(path_or_name_model)
    if path_model.exists():
        path_config = path_model.joinpath("config.json")
        path_model = path_model.joinpath("model.onnx")
    else:
        items: list[str] = path_or_name_model.split("/")
        assert len(items) >= 2
        repo_id: str = "/".join(items[:2])
        path_model = hf_hub_download(
            repo_id=repo_id,
            filename="/".join(items[2:]) + f"/{MODEL_FILE_NAME}",
        )
        path_config = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename="/".join(items[2:]) + f"/{CONFIG_FILE_NAME}",
            )
        )
    config: ImageTaggerConfig = ImageTaggerConfig.parse_file(path_config)

    providers: list[str] = ["CPUExecutionProvider"]
    if not force_cpu and "CUDAExecutionProvider" in rt.get_available_providers():
        providers = ["CUDAExecutionProvider"]

    model: rt.InferenceSession = rt.InferenceSession(
        path_model,
        providers=providers,
    )

    image_paths = list(glob_images(path_in, "**/*"))

    rconv = ResultConverter(path_or_name_model=path_or_name_model)
    is_json: bool = ext.lower() == "json"

    thresholds: dict[int, float]
    if threshold_str is None:
        if is_json:
            thresholds = config.thresholds_for_json
        else:
            thresholds = config.thresholds_for_txt
    else:
        thresholds = json.loads(threshold_str)

    def run_batch(path_imgs):
        imgs = np.array([im for _, im in path_imgs])

        input_name = model.get_inputs()[0].name
        probs = model.run(None, {input_name: imgs})[0]

        for (image_path, _), prob in zip(path_imgs, probs):
            result = rconv(
                prob=prob,
                thresholds=thresholds,
            )
            my_path_out: Path = image_path.parent
            if path_out is not None:
                rel: Path = image_path.parent.relative_to(path_in)
                my_path_out = path_out.joinpath(rel)
                my_path_out.mkdir(exist_ok=True, parents=True)
            with my_path_out.joinpath(image_path.stem + f".{ext}").open("w") as outf:
                if is_json:
                    outf.write(result.json(indent=4, ensure_ascii=False))
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
    for data_entry in tqdm(data, smoothing=0.0):
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


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        required=True,
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
        default="shirayu/img2tags/SmilingWolf__wd-v1-4-convnext-tagger-v2",
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
    )
    parser.add_argument(
        "--thresholds",
        "--th",
        type=str,
    )
    parser.add_argument("--cpu", action="store_true", help="Force to use CPU")

    return parser.parse_args()


def main() -> None:
    opts = get_opts()
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
