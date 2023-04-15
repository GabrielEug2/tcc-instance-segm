from pathlib import Path

from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import Metadata
from detectron2.structures import Instances, Boxes
import torch
import numpy as np
import cv2
import conversions_lib

def plot(anns_or_preds: dict, img_file: Path, out_file: Path):
	"""Plots the annotations or predictions on the image.

	Args:
		anns_or_preds (list): list of annotations or predictions in the format:
			{"classname": string,
			 "confidence": float, between 0 and 1,
			 "mask": binary or compact RLE,
			 "bbox": [x1, y1, x2, y2]}
		img_file (Path): path to the image the annotations will be plotted on
		out_file (Path): path to the output image

	Returns:
		np.ndarray: imagem no formato BGR
	"""
	classnames, scores, masks, boxes = [], [], [], []
	for ann in anns_or_preds:
		classnames.append(ann['classname'])
		scores.append(ann['confidence'])
		boxes.append(ann['bbox'])

		mask = ann['mask']
		if type(mask) == dict: # RLE
			masks.append(conversions_lib.rle_to_bin_mask(mask))
		else:
			masks.append(mask)
	
	class_list = set(classnames)
	metadata = Metadata()
	metadata.set(thing_classes = class_list)
	class_ids = []
	for classname in classnames:
		class_ids.append(class_list.index(classname))

	img = cv2.imread(str(img_file))
	h, w, _ = img.shape
	instances = Instances((h, w))
	instances.pred_classes = torch.tensor(class_ids, dtype=torch.int)
	instances.scores = torch.tensor(scores, dtype=torch.float)
	instances.pred_masks = torch.tensor(np.array(masks), dtype=torch.bool)
	instances.pred_boxes = Boxes(torch.tensor(boxes, dtype=torch.float))

	v = Visualizer(img, metadata)
	vis_out = v.draw_instance_predictions(instances)
	out_img = vis_out.get_image()

	cv2.imwrite(str(out_file), out_img)