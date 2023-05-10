from collections import defaultdict
import json
from pathlib import Path
from typing import Generator

from . import mask_conversions

class Predictions:
	"""Functions to work with predictions in segm_lib format.
	See segm_lib.inference.pred_manager.save() for more details"""

	def __init__(self, pred_dir: Path):
		if not pred_dir.exists():
			raise FileNotFoundError(f"Dir not found {str(pred_dir)}")

		self.root_dir = pred_dir

	def get_model_names(self):
		model_names = [f.stem for f in self.root_dir.glob('*') if f.is_dir()]

		return model_names

	def load(self, img_file_name: str, model_name: str) -> dict:
		pred_file = self.root_dir / model_name / f"{img_file_name}.json"
		try:
			predictions = self._load_from_file(pred_file)
		except FileNotFoundError:
			return []

		return predictions

	def _load_from_file(self, pred_file):
		with pred_file.open('r') as f:
			predictions = json.load(f)

		for pred in predictions:
			pred['mask'] = mask_conversions.rle_to_bin_mask(pred['mask'])

		# Could test keys and values too, but whatever

		return predictions

	def get_n_images_with_predictions(self, model_name: str) -> int:
		n_images = 0
		for pred_file in self._pred_files_for_model(model_name):
			predictions = self._load_from_file(pred_file)
			if len(predictions) >= 1:
				n_images += 1
		return n_images

	def _pred_files_for_model(self, model_name: str) -> Generator:
		return (self.root_dir / model_name).glob('*')

	def get_n_objects(self, model_name: str) -> int:
		count = 0
		for pred_file in self._pred_files_for_model(model_name):
			predictions = self._load_from_file(pred_file)

			for pred in predictions:
				count += 1
		return count
	
	def class_distribution(self, model_name: str) -> dict[str, int]:
		if hasattr(self, '_computed_class_dists') and model_name in self._computed_class_dists:
			return self._computed_class_dists[model_name]
		
		class_dist = defaultdict(lambda: 0)
		for pred_file in self._pred_files_for_model(model_name):
			predictions = self._load_from_file(pred_file)

			for pred in predictions:
				class_dist[pred['classname']] += 1
		class_dist = dict(class_dist)

		if not hasattr(self, '_computed_class_dists'):
			self._computed_class_dists = {}
		
		self._computed_class_dists[model_name] = class_dist
		return class_dist

	def get_classnames(self, model_name: str) -> list[str]:
		return self.class_distribution(model_name).keys()

	def filter(self, model_name: str, classes: list[str], out_dir: Path):
		out_dir = out_dir / model_name
		out_dir.mkdir(parents=True, exist_ok=True)

		for pred_file in self._pred_files_for_model(model_name):
			predictions = self._load_from_file(pred_file)

			filtered_preds = []
			for pred in predictions:
				if pred['classname'] in classes:
					filtered_preds.append(pred)
		
			for pred in filtered_preds:
				pred['mask'] = mask_conversions.bin_mask_to_rle(pred['mask'])

			with (out_dir / pred_file.name).open('w') as f:
				json.dump(filtered_preds, f)

	def to_coco_format(self, model_name: str, img_map: dict, classmap: dict, out_file: Path):
		coco_formatted_data = []
		for img_file_name in img_map:
			predictions = self.load(img_file_name, model_name)

			for pred in predictions:
				classname = pred['classname']
				if classname not in classmap:
					# would have to give a new id, but since there's no annotations
					# it doesn't make sense to keep them in my case (I only use
					# this to evaluate)
					continue

				formatted_pred = {
					"image_id": img_map[img_file_name],
					"category_id": classmap[classname],
					"segmentation": mask_conversions.bin_mask_to_rle(pred['mask']),
					"score": pred['confidence']
				}
				coco_formatted_data.append(formatted_pred)

		out_file.parent.mkdir(parents=True, exist_ok=True)
		with out_file.open('w') as f:
			json.dump(coco_formatted_data, f)