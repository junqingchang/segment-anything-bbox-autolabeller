# Segment Anything BBox Labeller
A bbox labeller built on [facebookresearch's segment anything](https://github.com/facebookresearch/segment-anything)

![sample image](https://github.com/junqingchang/segment-anything-bbox-autolabeller/blob/main/assets/sample.png?raw=true)

## Directory Structure
```
model/
    <segment anything models goes here>
.gitignore
auto_mask.py
README.md
requirements.txt
ui.py
```

## Run Interface
```
$ python ui.py
```

Additional parameters can be added to change the model path, model type, and output path using `-o <output-dir> -m <model-path> -t <model-type>` or `--output_dir <output-dir> --model <model-path> --model_type <model-type>` 