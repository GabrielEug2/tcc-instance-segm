from collections import defaultdict
from pathlib import Path

import cv2
import torch
from detectron2.data.catalog import Metadata
from detectron2.structures import Boxes, Instances
from detectron2.utils.visualizer import Visualizer

from ..core import mask_conversions
from ..core.structures import Annotation, Prediction
from .abstract_plot_lib import AbstractPlotLib


class DetectronPlotLib(AbstractPlotLib):
	lib_name = 'detectron'

	def __init__(self):
		pass

	def _plot(self, anns_or_preds: list[Annotation]|list[Prediction], img_file: Path, out_file: Path):
		classnames, scores, masks, boxes = [], [], [], []
		for ann_or_pred in anns_or_preds:
			classnames.append(ann_or_pred.classname)
			scores.append(ann_or_pred.confidence if hasattr(ann_or_pred, 'confidence') else 1.0)

			bin_mask = mask_conversions.rle_to_bin_mask(ann_or_pred.mask)
			masks.append(bin_mask)

			boxes.append(ann_or_pred.bbox)
		
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

	def _plot_individual_masks(self, anns_or_preds: list[Annotation]|list[Prediction], img_file: Path, out_dir: Path):
		count_per_class = defaultdict(lambda: 0)

		for ann_or_pred in anns_or_preds:
			classname = ann_or_pred.classname
			count_per_class[classname] += 1
			out_file_name = f"{classname}_{count_per_class[classname]}.jpg"

			plotted_mask_file = out_dir / f'{out_file_name}.jpg'
			self._plot([ann_or_pred], img_file, plotted_mask_file)