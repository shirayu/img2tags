
# img2tags

[![PyPI version](https://badge.fury.io/py/img2tags.svg)](https://badge.fury.io/py/img2tags)
[![Python Versions](https://img.shields.io/pypi/pyversions/img2tags.svg)](https://pypi.org/project/img2tags/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://pepy.tech/badge/img2tags/week)](https://pepy.tech/project/img2tags)

[![CI](https://github.com/shirayu/img2tags/actions/workflows/ci.yml/badge.svg)](https://github.com/shirayu/img2tags/actions/workflows/ci.yml)
[![Typos](https://github.com/shirayu/img2tags/actions/workflows/typos.yml/badge.svg)](https://github.com/shirayu/img2tags/actions/workflows/typos.yml)

## Setup

```bash
# For CPU
pip install 'img2tags[cpu]'

# For GPU
pip install 'img2tags[gpu]'
```

## Run

```bash
# Generate .txt file
img2tags -i input_dir

# Generate .json file in another directory with CPU
img2tags --model shirayu/img2tags/SmilingWolf__wd-v1-4-moat-tagger-v2 \
    --batch 4 \
    --ext json \
    --cpu \
    -i input_dir \
    -o output_dir
```

## LICENSE

Apache 2.0

Some codes are based on codes in the following project.

- <https://github.com/kohya-ss/sd-scripts>

## References

- [Convert onnx models](https://github.com/onnx/onnx/issues/2182#issuecomment-881752539)
