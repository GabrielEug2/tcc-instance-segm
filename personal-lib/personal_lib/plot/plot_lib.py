from pathlib import Path

from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import Metadata
from detectron2.structures import Instances, Boxes
import cv2
import torch
import numpy as np

def plot(anns_or_preds: dict, img_file: Path, out_file: Path):
	"""Plots the annotations or predictions on the image.

	Args:
		anns_or_preds (list): list of annotations or predictions in the format:
			{"classname": string,
			 "confidence": float, between 0 and 1,
			 "mask": torch.BoolTensor,
			 "bbox": [x1, y1, x2, y2]}
		img_file (Path): path to the image the annotations will be plotted on
		out_file (Path): path to the output image
	"""
	classnames, scores, masks, boxes = [], [], [], []
	for pred in anns_or_preds:
		classnames.append(pred['classname'])
		scores.append(pred['confidence'])
		masks.append(pred['mask'])
		boxes.append(pred['bbox'])
	
	class_list = list(set(classnames))
	class_ids = []
	for classname in classnames:
		class_ids.append(class_list.index(classname))
	metadata = Metadata()
	metadata.set(thing_classes = class_list)

	img = cv2.imread(str(img_file))
	h, w, _ = img.shape
	instances = Instances((h, w))
	instances.pred_classes = torch.tensor(class_ids, dtype=torch.int)
	instances.scores = torch.tensor(scores, dtype=torch.float)
	instances.pred_masks = torch.stack(masks) if masks else torch.tensor(masks)
	instances.pred_boxes = Boxes(torch.tensor(boxes, dtype=torch.float))

	v = Visualizer(img, metadata)
	vis_out = v.draw_instance_predictions(instances)
	out_img = vis_out.get_image()

	cv2.imwrite(str(out_file), out_img)

def plot_individual_masks(anns_or_preds: dict, out_dir: Path, img_file: Path):
	out_dir.mkdir(exist_ok=True)

	count_per_class = {}
	for pred in anns_or_preds:
		classname = pred['classname']
		mask = pred['mask']

		count_per_class[classname] = count_per_class.get(classname, 0) + 1
		out_file_basename = f"{classname}_{count_per_class[classname]}"

		bin_mask_file = out_dir / f"{out_file_basename}_bin.jpg"
		bin_mask_np = mask.numpy().astype(np.uint8) * 255
		cv2.imwrite(str(bin_mask_file), bin_mask_np)

		plotted_mask_file = out_dir / f"{out_file_basename}_plot.jpg"
		plot([pred], img_file, plotted_mask_file)