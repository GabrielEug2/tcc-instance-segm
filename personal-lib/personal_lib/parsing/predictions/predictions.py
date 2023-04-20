"""Functions to worh with predictions in personal_lib format.

See personal_lib.inference.predictors.base_predictor for more details"""

import json
from pathlib import Path

from personal_lib.parsing.common import mask_conversions

class Predictions():
	def __init__(self, predictions):
		self.predictions = predictions

	def save(self, out_file):
		for pred in self.predictions:
			pred['mask'] = mask_conversions.bin_mask_to_rle(pred['mask'])

		with out_file.open('w') as f:
			json.dump(self.predictions, f)

	@classmethod
	def load_from_file(self, pred_file: Path):
		if not pred_file.exists():
			raise FileNotFoundError(str(pred_file))

		try:
			with pred_file.open('r') as f:
				predictions = json.load(f)

			for pred in predictions:
				pred['mask'] = mask_conversions.rle_to_bin_mask(pred['mask'])
		except json.JSONDecodeError as e:
			raise ValueError(f"File \"{str(pred_file)}\" does not follow the required format") from e

		return Predictions(predictions)

	def class_distribution(self) -> dict:
		class_dist = {}
		for pred in self.predictions:
			classname = pred['classname']
			class_dist[classname] = class_dist.get(classname, 0) + 1

		return class_dist