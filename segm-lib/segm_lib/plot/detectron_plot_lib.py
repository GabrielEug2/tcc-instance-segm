from pathlib import Path

from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import Metadata
from detectron2.structures import Instances, Boxes
import cv2
import torch
import numpy as np

from .abstract_plot_lib import AbstractPlotLib

class DetectronPlotLib(AbstractPlotLib):
	def __init__(self):
		pass

	def plot(self, anns_or_preds: list, img_file: Path, out_file: Path):
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

		boxes_for_api = []
		for box in boxes:
			# pra essa API do Detectron precisa estar em [x1,y1,x2,y2]
			x1, y1, w, h = box
			x2 = x1 + w
			y2 = y1 + h
			boxes_for_api.append([x1, y1, x2, y2])

		img = cv2.imread(str(img_file))
		h, w, _ = img.shape
		instances = Instances((h, w))
		instances.pred_classes = torch.tensor(class_ids, dtype=torch.int)
		instances.scores = torch.tensor(scores, dtype=torch.float)
		instances.pred_masks = torch.stack(masks) if len(masks) >= 1 else torch.tensor([])
		instances.pred_boxes = Boxes(torch.tensor(boxes_for_api, dtype=torch.float))

		v = Visualizer(img, metadata)
		vis_out = v.draw_instance_predictions(instances)
		out_img = vis_out.get_image()

		out_file.parent.mkdir(parents=True, exist_ok=True)
		cv2.imwrite(str(out_file), out_img)

	def plot_individual_masks(self, anns_or_preds: list, out_dir: Path, img_file: Path):
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
			self.plot([pred], img_file, plotted_mask_file)