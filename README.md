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

## Download Model Checkpoint
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Run Interface
```
$ python ui.py
```

Additional parameters can be added to change the model path, model type, and output path using `-o <output-dir> -m <model-path> -t <model-type>` or `--output_dir <output-dir> --model <model-path> --model_type <model-type>` 

# Grounded SAM
```
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git

pip install -e Grounded-Segment-Anything/GroundingDINO

wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```
