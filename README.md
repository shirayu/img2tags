
# img2tags

[![PyPI version](https://badge.fury.io/py/img2tags.svg)](https://badge.fury.io/py/img2tags)
[![Python Versions](https://img.shields.io/pypi/pyversions/img2tags.svg)](https://pypi.org/project/img2tags/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://pepy.tech/badge/img2tags/week)](https://pepy.tech/project/img2tags)

[![CI](https://github.com/shirayu/img2tags/actions/workflows/ci.yml/badge.svg)](https://github.com/shirayu/img2tags/actions/workflows/ci.yml)
[![Typos](https://github.com/shirayu/img2tags/actions/workflows/typos.yml/badge.svg)](https://github.com/shirayu/img2tags/actions/workflows/typos.yml)

Tag images by using [ONNX Runtime](https://onnxruntime.ai/).
The current default model is [SmilingWolf/wd-vit-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-vit-large-tagger-v3).
It works fine for anime images and with batch sizes greater than 1.

## Setup

```bash
# For CPU
pip install -U 'img2tags[cpu]'

# For GPU
pip install -U 'img2tags[gpu]'

# If you use CUDA 12.2, the following packages may be needed
sudo apt install libcufft10 libcudart11.0
```

## Run

```bash
# Generate .txt file
img2tags -i input_dir

# Generate .json file in another directory with CPU
#   --offline: Do not fetch model files (You must have downloaded that model)
img2tags --ext json \
    --cpu \
    -i input_dir \
    -o output_dir \
    --offline

# Output to a single file (JSONL)
img2tags --ext jsonl \
    --cpu \
    -i input_dir \
    -o output_filename.jsonl \
    --offline

# You can use filelist
img2tags --ext jsonl \
    --cpu \
    -i target_file_list.txt \
    -o output_filename.jsonl \
    --offline
```

You can set thresholds in JSON format like ``--th {"0": 0.3, "4":0.2, "9":0.3}``.

## LICENSE

Apache 2.0

Some codes are based on codes in the following project.

- <https://github.com/kohya-ss/sd-scripts>

## References

- [Convert onnx models to change batch size](https://github.com/onnx/onnx/issues/2182#issuecomment-881752539)
