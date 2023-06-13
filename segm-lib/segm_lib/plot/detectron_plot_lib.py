from collections import defaultdict
from pathlib import Path

import cv2
import torch
from detectron2.data.catalog import Metadata
from detectron2.structures import Boxes, Instances
from detectron2.utils.visualizer import Visualizer, ColorMode

from segm_lib.core import mask_conversions
from segm_lib.core.structures import Annotation, Prediction
from .plot_lib import PlotLib


class DetectronPlotLib(PlotLib):
	def __init__(self):
		pass

	def plot(self, anns_or_preds: list[Annotation|Prediction], img_file: Path, out_file: Path,
			colors: dict[str: [int, int, int]] = None
			):
		"""Plots the annotations or predictions on the image.

		Args:
			anns_or_preds (list): list of annotations or predictions to plot.
			img_file (Path): path to the image the annotations will be plotted on.
			out_file (Path): path to the output image. If it doesn't exist, it will
				be created. If it does, it will be overriten.
			colors (dict, optional): list of colors in RGB to use for each class. If not
				specified, random colors will be used.
		"""
		classnames, scores, masks, boxes = [], [], [], []
		for ann_or_pred in anns_or_preds:
			classnames.append(ann_or_pred.classname)
			scores.append(ann_or_pred.confidence if hasattr(ann_or_pred, 'confidence') else 1.0)

			bin_mask = mask_conversions.rle_to_bin_mask(ann_or_pred.mask)
			masks.append(bin_mask)

			# pra essa API do Detectron precisa estar em [x1,y1,x2,y2]
			x1, y1, w, h = ann_or_pred.bbox
			x2 = x1 + w
			y2 = y1 + h
			boxes.append([x1, y1, x2, y2])
		
		class_list = list(set(classnames))
		class_ids = []
		for classname in classnames:
			class_ids.append(class_list.index(classname))
		metadata = Metadata()
		metadata.set(thing_classes = class_list)
		if colors:
			colors_for_api = []
			for classname in class_list:
				colors_for_api.append(colors[classname])
			metadata.set(thing_colors = colors_for_api)

		img = cv2.imread(str(img_file))
		h, w, _ = img.shape
		instances = Instances((h, w))
		instances.pred_classes = torch.tensor(class_ids, dtype=torch.int)
		instances.scores = torch.tensor(scores, dtype=torch.float)
		instances.pred_masks = torch.stack(masks) if len(masks) >= 1 else torch.tensor([])
		instances.pred_boxes = Boxes(torch.tensor(boxes, dtype=torch.float))

		if colors is None:			
			v = Visualizer(img, metadata)
		else:
			v = Visualizer(img, metadata, instance_mode=ColorMode.SEGMENTATION)
		vis_out = v.draw_instance_predictions(instances)
		out_img = vis_out.get_image()

		out_file.parent.mkdir(parents=True, exist_ok=True)
		cv2.imwrite(str(out_file), out_img)

	def plot_individual_masks(self, anns_or_preds: list[Annotation|Prediction], img_file: Path, out_dir: Path):
		count_per_class = defaultdict(lambda: 0)

		for ann_or_pred in anns_or_preds:
			classname = ann_or_pred.classname
			count_per_class[classname] += 1
			out_file_name = f"{classname}_{count_per_class[classname]}.jpg"

			plotted_mask_file = out_dir / f'{out_file_name}.jpg'
			self.plot([ann_or_pred], img_file, plotted_mask_file)