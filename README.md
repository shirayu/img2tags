
# img2tags

[![PyPI version](https://badge.fury.io/py/img2tags.svg)](https://badge.fury.io/py/img2tags)
[![Python Versions](https://img.shields.io/pypi/pyversions/img2tags.svg)](https://pypi.org/project/img2tags/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://pepy.tech/badge/img2tags/week)](https://pepy.tech/project/img2tags)

[![CI](https://github.com/shirayu/img2tags/actions/workflows/ci.yml/badge.svg)](https://github.com/shirayu/img2tags/actions/workflows/ci.yml)
[![Typos](https://github.com/shirayu/img2tags/actions/workflows/typos.yml/badge.svg)](https://github.com/shirayu/img2tags/actions/workflows/typos.yml)

Tag images by using [ONNX Runtime](https://onnxruntime.ai/).
The current default model is the converted ``WD 1.4 ConvNext Tagger V2`` on <https://huggingface.co/shirayu/img2tags>.
It works fine for anime images.

## Setup

```bash
# For CPU
pip install -U 'img2tags[cpu]'

# For GPU
pip install -U 'img2tags[gpu]'
```

## Run

```bash
# Generate .txt file
img2tags -i input_dir

# Generate .json file in another directory with CPU
img2tags --model shirayu/img2tags/SmilingWolf__wd-v1-4-convnext-tagger-v2 \
    --batch 4 \
    --ext json \
    --cpu \
    -i input_dir \
    -o output_dir
```

The values of default thresholds are stored in ``config.json`` in each model like [this](https://huggingface.co/shirayu/img2tags/blob/main/SmilingWolf__wd-v1-4-convnext-tagger-v2/config.json).
You can set values in JSON format like ``--th {"0": 0.3, "4":0.2, "9":0.3}``.

## LICENSE

Apache 2.0

Some codes are based on codes in the following project.

- <https://github.com/kohya-ss/sd-scripts>

## References

- [Convert onnx models to change batch size](https://github.com/onnx/onnx/issues/2182#issuecomment-881752539)
