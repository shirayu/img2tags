#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def operation(
    *,
    path_in: Path,
    path_out: Path,
) -> None:
    with path_in.open() as inf, path_out.open("w") as outf:
        for line in inf:
            d = json.loads(line)
            year: int = int(d["created_at"][:4])
            if year < 2020:
                continue
            image_width: int = d["image_width"]
            image_height: int = d["image_height"]

            small: int = min(image_width, image_height)
            w_ration: float = image_width / small
            h_ration: float = image_height / small

            outf.write(f"{image_width}\t{image_height}\t{w_ration:.2}\t{h_ration:.2}\n")


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, default="/dev/stdin", required=False)
    oparser.add_argument("--output", "-o", type=Path, default="/dev/stdout", required=False)
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()
    operation(
        path_in=opts.input,
        path_out=opts.output,
    )


if __name__ == "__main__":
    main()
