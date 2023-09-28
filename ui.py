import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import argparse
from segment_anything import sam_model_registry, SamPredictor
from torchvision.ops import masks_to_boxes


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(
    prog="SegmentAnythingBBoxLabeller",
    description="Generate mask, metadata, overlay for iamge",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    default="sample_outputs",
    help="output directory for generated mask and metadata",
)
parser.add_argument(
    "-m", "--model", type=str, default="models/sam_vit_h_4b8939.pth", help="model path"
)
parser.add_argument(
    "-t",
    "--model_type",
    type=str,
    default="vit_h",
    help="type of model defined by segment anything",
)

args = parser.parse_args()


class SegmentAnythingBBoxLabeller:
    def __init__(self, root):
        self.root = root
        self.root.title("SegmentAnythingBBoxLabeller")
        self.root.geometry("300x100")

        # pack blank to make it neater
        self.blank = tk.Label(root, text="")
        self.blank.pack()

        # Button to load an image
        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        # Button to call label image
        self.label_button = tk.Button(
            root, text="Label Image", command=self.label_image
        )
        self.label_button.pack()

        # Label to display coordinates
        self.coordinates_label = tk.Label(
            root, text="Click on the image to detect coordinates"
        )
        self.coordinates_label.pack()

        # Segment Anything model loading
        self.sam = sam_model_registry[args.model_type](checkpoint=args.model)
        self.sam.to(device=DEVICE)
        self.predictor = SamPredictor(self.sam)

        self.point_coords = []
        self.point_labels = []
        self.image = None
        self.output_image = None

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.bmp *.jpeg")]
        )
        if file_path:
            self.image = cv2.imread(file_path)
            self.clear_inputs()
            self.display_image()

    def display_image(self):
        if self.image is not None:
            # Show the image using cv2.imshow in a separate window
            cv2.imshow("Image", self.image)
            cv2.setMouseCallback("Image", self.detect_coordinates)

    def clear_inputs(self):
        self.point_coords = []
        self.point_labels = []

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def label_image(self):
        cv2.destroyAllWindows()
        self.load_button["state"] = "disabled"
        self.label_button["state"] = "disabled"
        if self.image is not None:
            tmp_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(tmp_image)
            masks, scores, logits = self.predictor.predict(
                point_coords=np.array(self.point_coords),
                point_labels=np.array(self.point_labels),
                multimask_output=False,
            )
            for i, (mask, score) in enumerate(zip(masks, scores)):
                plt.figure(figsize=(10, 10))
                plt.imshow(tmp_image)
                self.show_mask(mask, plt.gca())
            boxes = masks_to_boxes(torch.Tensor(masks))
            ax = plt.gca()
            for box in boxes:
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)

            box_coords = [" ".join(map(str, map(int, box.tolist()))) for box in boxes]
            with open(os.path.join(args.output_dir, "bbox_points"), "w") as f:
                f.writelines(box_coords)

            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis("off")
            fig = plt.gcf()
            fig.savefig(os.path.join(args.output_dir, "output_segment.png"))
            fig.show()

        self.load_button["state"] = "normal"
        self.label_button["state"] = "normal"

    def detect_coordinates(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.coordinates_label.config(text=f"Coordinates: ({x}, {y})")
            self.point_coords.append([x, y])
            self.point_labels.append(1)

            # Draw a marker on the image
            marker_color = (0, 0, 255)  # BGR color (red)
            marker_radius = 5
            cv2.circle(
                self.image, (x, y), marker_radius, marker_color, -1
            )  # -1 fills the circle

            # Redraw the updated image
            cv2.imshow("Image", self.image)


if __name__ == "__main__":
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    root = tk.Tk()
    app = SegmentAnythingBBoxLabeller(root)
    root.mainloop()
