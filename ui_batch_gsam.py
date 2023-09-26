import argparse
import os
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import supervision as sv
import torch
import torchvision
from groundingdino.util.inference import Model, load_model, load_image, predict, annotate
from segment_anything import sam_model_registry, SamPredictor

DEVICE_STRING = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE_STRING)

parser = argparse.ArgumentParser(prog='GSAMBatch', description='auto segmentation based on prompt')
parser.add_argument('-gm', '--groundingdino_model',  type=str, default='models/groundingdino_swint_ogc.pth', help='model path')
parser.add_argument('-gc', '--groundingdino_model_config', type=str, default='configs/groundingdino.py', help='config path')
parser.add_argument('-sm', '--sam_model', type=str, default='models/sam_vit_h_4b8939.pth', help='sam model path')
parser.add_argument('-st', '--sam_model_type', type=str, default='vit_h', help='sam model type')
parser.add_argument('-o', '--output_dir', type=str, default='sample_outputs', help='output directory')
parser.add_argument('-p', '--prompt', type=str, default='ships', help='search string delimited with commas')

args = parser.parse_args()

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=args.groundingdino_model_config,
                             model_checkpoint_path=args.groundingdino_model,
                             device=DEVICE_STRING)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_model)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)


class GSAMBatch:
  def __init__(self, root):
    self.root = root
    self.root.title("GSAMBatch")
    self.root.geometry("400x150")

    # pack blank to make it neater
    self.blank = tk.Label(root, text="")
    self.blank.pack()

    # button to load images
    self.load_button = tk.Button(root, text="Load Images", command=self.load_images)
    self.load_button.pack()

    # button to label images
    self.label_button = tk.Button(root, text="Label Images", command=self.label_image_gsam)
    self.label_button.pack()

    # data
    self.filepath_input = None
    self.filepaths = []

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

  def load_images(self):
    file_dir = filedialog.askdirectory()
    if not file_dir:
      return

    self.filepath_input = file_dir
    for root, dirs, files in os.walk(file_dir):
      for file in files:
        if self.is_supported_filetype(file):
          self.filepaths.append(os.path.join(root, file))

  def label_image(self):
    print('Starting segmentation...')

    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25

    for filepath in self.filepaths:
      image_source, image = load_image(filepath)
      model = load_model(args.config, args.model)
      boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=args.prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE_STRING,
      )

      annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
      outpath = filepath.replace(self.filepath_input, args.output_dir)
      os.makedirs(os.path.dirname(outpath), exist_ok=True)
      cv2.imwrite(outpath, annotated_frame)

    print('Finished!')

  def label_image_gsam(self):
    print('Starting segmentation...')

    BOX_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.8

    CLASSES = args.prompt.split(",")

    # Predict classes and hyper-param for GroundingDINO
    for filepath in self.filepaths:

      # load image
      image = cv2.imread(filepath)

      # detect objects
      detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=BOX_THRESHOLD
      )

      # NMS post process
      nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy),
        torch.from_numpy(detections.confidence),
        NMS_THRESHOLD
      ).numpy().tolist()

      detections.xyxy = detections.xyxy[nms_idx]
      detections.confidence = detections.confidence[nms_idx]
      detections.class_id = detections.class_id[nms_idx]

      # Prompting SAM with detected boxes
      def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
          masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
          )
          index = np.argmax(scores)
          result_masks.append(masks[index])
        return np.array(result_masks)

      # convert detections to masks
      detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
      )

      # annotate image with detections
      box_annotator = sv.BoxAnnotator()
      mask_annotator = sv.MaskAnnotator()
      labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections]
      annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
      annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

      # save the annotated grounded-sam image
      outpath = filepath.replace(self.filepath_input, args.output_dir)
      os.makedirs(os.path.dirname(outpath), exist_ok=True)
      cv2.imwrite(outpath, annotated_image)

    print('Finished!')


if __name__ == "__main__":
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  root = tk.Tk()
  app = GSAMBatch(root)
  app.root.mainloop()
