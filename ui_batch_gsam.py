import torch
import tkinter as tk
from tkinter import filedialog

from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os
import argparse

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(prog='GSAMBatch', description='auto segmentation based on prompt')
parser.add_argument('-o', '--output_dir', type=str, default='sample_outputs', help='output directory')
parser.add_argument('-m', '--model',  type=str, default='models/groundingdino_swint_ogc.pth', help='model path')
parser.add_argument('-c', '--config', type=str, default='configs/gsam.py', help='config path')
parser.add_argument('-p', '--prompt', type=str, default='ships', help='objects to search for, delimited with fullstops')

args = parser.parse_args()


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
    self.label_button = tk.Button(root, text="Label Images", command=self.label_image)
    self.label_button.pack()

    # data
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

  def load_images(self):
    file_dir = filedialog.askdirectory()
    if not file_dir:
      return

    dir, dirs, files = next(os.walk(file_dir))
    self.filepaths = [
      os.path.join(dir, file) for file in files if self.is_supported_filetype(file)]

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
        device=DEVICE,
      )

      annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
      cv2.imwrite(os.path.join(args.output_dir, os.path.basename(filepath)), annotated_frame)

    print('Finished!')


if __name__ == "__main__":
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  root = tk.Tk()
  app = GSAMBatch(root)
  app.root.mainloop()
