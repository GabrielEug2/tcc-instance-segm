"""Functions to work with predictions in personal_lib format.
See personal_lib.inference.predictors.base_predictor for more details"""

import json
from pathlib import Path

from personal_lib.parsing.common import mask_conversions

class PredictionManager:
	def __init__(self, root_dir: Path|None):
		if not root_dir.exists():
			root_dir.mkdir()

		self.root_dir = root_dir

	def save(self, preds: dict, img_file: Path, model_name: str):
		for pred in preds:
			pred['mask'] = mask_conversions.bin_mask_to_rle(pred['mask'])

		out_file = self.root_dir / img_file.stem / f"{model_name}_preds.json"
		out_file.mkdir(parents=True, exist_ok=True)
		with out_file.open('w') as f:
			json.dump(preds, f)

	def load(self, img_file: Path, model_name: str) -> dict:
		pred_file = self.root_dir / img_file.stem / f"{model_name}_preds.json"
		predictions = self._load_from_file(pred_file)
		return predictions

	def _load_from_file(self, pred_file):
		with pred_file.open('r') as f:
			predictions = json.load(f)

		for pred in predictions:
			pred['mask'] = mask_conversions.rle_to_bin_mask(pred['mask'])

		# Could test keys and values too, but whatever

		return predictions

	def get_model_names(self):
		model_names = set()
		all_files = self.root_dir.rglob('*_preds.json')

		for file in all_files:
			model_used = file.name.split('_')[0]
			if model_used not in model_names:
				model_names.add(model_used)

		return model_names

	def class_distribution(self, model_name: str) -> dict:
		relevant_files = self._get_files_for_model(model_name)

		class_dist = {}
		for file in relevant_files:
			predictions = self._load_from_file(file)
			for pred in predictions:
				classname = pred['classname']
				class_dist[classname] = class_dist.get(classname, 0) + 1

		return class_dist
	
	def _get_files_for_model(self, model_name):
		all_files = self.root_dir.rglob('*_preds.json')
		relevant_files = (f for f in all_files if f.name.startswith(model_name))
		return relevant_files