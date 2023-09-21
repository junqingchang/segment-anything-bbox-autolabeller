import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import argparse
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(
                    prog='AutoMask',
                    description='Generate mask, metadata, overlay for iamge')
parser.add_argument('-i', '--input_image', type=str, default='sample_inputs/truck.jpg', help='input image to run')
parser.add_argument('-o', '--output_dir', type=str, default='sample_outputs', help='output directory for generated mask and metadata')
parser.add_argument('-m', '--model', type=str, default='model/sam_vit_h_4b8939.pth', help='model path')
parser.add_argument('-t', '--model_type', type=str, default='vit_h', help='type of model defined by segment anything')
args = parser.parse_args()


def write_masks_to_folder(masks , path):
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h" 
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))
    return


def generate_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def main():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    image = cv2.imread(args.input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam = sam_model_registry[args.model_type](checkpoint=args.model)
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam
        )
    masks = mask_generator.generate(image)
    write_masks_to_folder(masks, args.output_dir)
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    generate_anns(masks)
    plt.axis('off')
    plt.savefig(os.path.join(args.output_dir, 'overlay.png'))


if __name__ == '__main__':
    main()
