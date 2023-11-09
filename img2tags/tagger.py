#!/usr/bin/env python3
import csv
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from pydantic import BaseModel


def trim_tag(tag_name: str) -> str:
    if len(tag_name) > 3:  # ignore emoji tags like >_< and ^_^
        return tag_name.replace("_", " ")
    return tag_name


class ImageTaggerConfig(BaseModel):
    image_size: int
    thresholds_for_json: dict[int, float]
    thresholds_for_txt: dict[int, float]


class ImageTagger:
    pass

    @staticmethod
    def preprocess(
        *,
        image: Image.Image,
        image_size: int,
    ):
        _image = np.array(image)
        _image = _image[:, :, ::-1]  # RGB->BGR

        # pad to square
        size = max(_image.shape[0:2])
        pad_x = size - _image.shape[1]
        pad_y = size - _image.shape[0]
        pad_l = pad_x // 2
        pad_t = pad_y // 2
        _image = np.pad(  # type: ignore
            _image,
            ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)),  # type: ignore
            mode="constant",
            constant_values=255,
        )

        interp = cv2.INTER_AREA if size > image_size else cv2.INTER_LANCZOS4
        _image = cv2.resize(_image, (image_size, image_size), interpolation=interp)

        return _image.astype(np.float32)


CSV_FILE: Final[str] = "selected_tags.csv"


class Result(BaseModel):
    tags: dict[int, dict[str, float]]


class ResultConverter:
    def __init__(
        self,
        *,
        path_or_name_model: str,
        filename: str = CSV_FILE,
    ):
        tags_path: Path
        if Path(path_or_name_model).exists():
            tags_path = Path(path_or_name_model).joinpath(filename)
        else:
            items: list[str] = path_or_name_model.split("/")
            assert len(items) >= 2

            myfilename: str
            if len(items) == 2:
                myfilename = filename
            else:
                myfilename = "/".join(items[2:]) + f"/{filename}"

            repo_id: str = "/".join(items[:2])

            tags_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=myfilename,
                )
            )

        self.tags: list[str] = []
        self.categories: list[int] = []
        with tags_path.open() as f:
            reader = csv.reader(f)
            header = next(reader)
            #             idx_tag :int = header.index("tag_id")
            idx_name: int = header.index("name")
            idx_category: int = header.index("category")

            for row in reader:
                self.categories.append(int(row[idx_category]))
                self.tags.append(row[idx_name])

    def __call__(
        self,
        *,
        prob,
        thresholds: dict[int, float],
    ) -> Result:
        ret: Result = Result(
            tags={},
        )
        for i, p in enumerate(prob):
            category: int = self.categories[i]
            if p > thresholds.get(category, 10000):
                if category not in ret.tags:
                    ret.tags[category] = {}
                ret.tags[category][self.tags[i]] = float(p)
        return ret

    ignore_catetories: list[int] = []
