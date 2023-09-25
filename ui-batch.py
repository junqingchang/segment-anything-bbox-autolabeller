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
from common import PredictorInput

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(prog='SegmentAnythingBBoxLabellerBatch', description='Generate mask, metadata, overlay for iamge')
parser.add_argument('-o', '--output_dir', type=str, default='sample_outputs', help='output directory for generated mask and metadata')
parser.add_argument('-m', '--model', type=str, default='model/sam_vit_h_4b8939.pth', help='model path')
parser.add_argument('-t', '--model_type', type=str, default='vit_h', help='type of model defined by segment anything')

args = parser.parse_args()


class SegmentAnythingBBoxLabellerBatch:
  def __init__(self, root):
    self.root = root
    self.root.title("SegmentAnythingBBoxLabellerBatch")
    self.root.geometry("400x150")

    # pack blank to make it neater
    self.blank = tk.Label(root, text="")
    self.blank.pack()

    # Button to load images
    self.load_button = tk.Button(root, text="Load Images", command=self.load_images)
    self.load_button.pack()

    # Button to load next image
    self.load_button = tk.Button(root, text="Next Image", command=self.load_next_image)
    self.load_button.pack()

    # Button to label images
    self.label_button = tk.Button(root, text="Label Images", command=self.label_image)
    self.label_button.pack()

    # Label to display coordinates
    self.coordinates_label = tk.Label(root, text="Click on the image to detect coordinates")
    self.coordinates_label.pack()

    # Segment Anything model loading
    self.sam = sam_model_registry[args.model_type](checkpoint=args.model)
    self.sam.to(device=DEVICE)
    self.predictor = SamPredictor(self.sam)
    self.predictor_inputs = []

    self.point_coords = []
    self.point_labels = []
    self.image = None
    self.image_filepath = None

    self.filepaths = None

  def is_supported_filetype(self, file):
    if file.endswith('.jpg'):
      return True
    if file.endswith('.png'):
      return True
    if file.endswith('.bmp'):
      return True
    if file.endswith('.jpeg'):
      return True

    return False

  def load_next_image(self):
    if self.image is not None:
      predictor_input = PredictorInput(self.image_filepath, self.point_coords, self.point_labels)
      self.predictor_inputs.append(predictor_input)

    next_image = next(self.filepaths)
    if next_image is not None:
      self.load_image(next_image)

  def load_images(self):
    file_dir = filedialog.askdirectory()
    if not file_dir:
      return

    dir, dirs, files = next(os.walk(file_dir))
    self.filepaths = (os.path.join(dir, file) for file in files if self.is_supported_filetype(file))
    self.load_next_image()

  def load_image(self, file_path):
    self.clear_inputs()
    self.image = cv2.imread(file_path)
    self.image_filepath = file_path
    self.display_image()

  def display_image(self):
    if self.image is not None:
      # Show the image using cv2.imshow in a separate window
      cv2.imshow("Image", self.image)
      cv2.setMouseCallback("Image", self.detect_coordinates)

  def clear_inputs(self):
    self.image = None
    self.image_filepath = None
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
    print('Starting segmentation...')
    cv2.destroyAllWindows()
    self.load_button['state'] = 'disabled'
    self.label_button['state'] = 'disabled'
    for model_input in self.predictor_inputs:
      outfile = os.path.basename(model_input.filepath)
      image = cv2.cvtColor(cv2.imread(model_input.filepath), cv2.COLOR_BGR2RGB)
      self.predictor.set_image(image)
      masks, scores, logits = self.predictor.predict(
        point_coords=np.array(model_input.point_coords),
        point_labels=np.array(model_input.point_labels),
        multimask_output=False,
      )
      for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        self.show_mask(mask, plt.gca())
      boxes = masks_to_boxes(torch.Tensor(masks))
      ax = plt.gca()
      for box in boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

      box_coords = [' '.join(map(str, map(int, box.tolist()))) for box in boxes]
      with open(os.path.join(args.output_dir, f'{outfile}.bbox-points.txt'), 'w') as f:
        f.writelines(box_coords)

      plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
      plt.axis('off')
      fig = plt.gcf()
      fig.savefig(os.path.join(args.output_dir, f'{outfile}'))
      # fig.show()

    self.load_button['state'] = 'normal'
    self.label_button['state'] = 'normal'
    print('Finished!')

  def detect_coordinates(self, event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
      self.coordinates_label.config(text=f"Coordinates: ({x}, {y})")
      self.point_coords.append([x, y])
      self.point_labels.append(1)

      # Draw a marker on the image
      marker_color = (0, 0, 255)  # BGR color (red)
      marker_radius = 5
      cv2.circle(self.image, (x, y), marker_radius, marker_color, -1)  # -1 fills the circle

      # Redraw the updated image
      cv2.imshow("Image", self.image)


if __name__ == "__main__":
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  root = tk.Tk()
  app = SegmentAnythingBBoxLabellerBatch(root)
  root.mainloop()
