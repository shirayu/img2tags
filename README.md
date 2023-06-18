
# img2tags

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
