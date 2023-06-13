from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from segm_lib.core import mask_conversions
from segm_lib.core.structures import Annotation, Prediction
from .plot_lib import PlotLib


class BinPlotLib(PlotLib):
	def __init__(self):
		pass

	def plot(self, anns_or_preds: list[Annotation|Prediction], img_file: Path, out_file: Path):
		# A bin image with all the anns/preds is not really useful to me
		return

	def plot_individual_masks(self, anns_or_preds: list[Annotation|Prediction], img_file: Path, out_dir: Path):
		out_dir.mkdir(parents=True, exist_ok=True)

		count_per_class = defaultdict(lambda: 0)
		for ann_or_pred in anns_or_preds:
			classname = ann_or_pred.classname
			count_per_class[classname] += 1
			out_file_name = f"{classname}_{count_per_class[classname]}.jpg"

			bin_mask = mask_conversions.rle_to_bin_mask(ann_or_pred.mask)
			
			bin_mask_file = out_dir / out_file_name
			bin_mask_np = bin_mask.numpy().astype(np.uint8) * 255
			cv2.imwrite(str(bin_mask_file), bin_mask_np)