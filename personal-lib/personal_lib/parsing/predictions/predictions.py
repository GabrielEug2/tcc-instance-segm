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

	def save(self, preds: dict, img_file: str, model_name: str):
		for pred in preds:
			pred['mask'] = mask_conversions.bin_mask_to_rle(pred['mask'])

		out_file = self.root_dir / img_file / f"{model_name}.json"
		out_file.parent.mkdir(parents=True, exist_ok=True)
		with out_file.open('w') as f:
			json.dump(preds, f)

	def load(self, img_file: str, model_name: str) -> dict:
		pred_file = self.root_dir / img_file / f"{model_name}.json"
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
		all_pred_files = self.root_dir.rglob('*.json')

		for file in all_pred_files:
			model_used = file.stem
			if model_used not in model_names:
				model_names.add(model_used)

		return model_names

	def class_distribution(self, model_name: str) -> dict:
		imgs_with_predictions_from_model = self._get_imgs_for_model(model_name)

		class_dist = {}
		for img_file in imgs_with_predictions_from_model:
			predictions = self.load(img_file, model_name)
			for pred in predictions:
				classname = pred['classname']
				class_dist[classname] = class_dist.get(classname, 0) + 1

		return class_dist
	
	def _get_imgs_for_model(self, model_name: str):
		imgs_with_predictions = (f for f in self.root_dir.glob('*') if f.is_dir())

		imgs_with_predictions_from_model = []
		for img_file in imgs_with_predictions:
			if Path(img_file / f"{model_name}.json").exists():
				imgs_with_predictions_from_model.append(img_file.stem)

		return imgs_with_predictions_from_model