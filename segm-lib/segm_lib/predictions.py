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

	def get_model_names(self) -> list[str]:
		return [f.stem for f in self.root_dir.glob('*') if f.is_dir()]

	def load(self, img_file_name: str, model_name: str) -> dict:
		pred_file = self.root_dir / model_name / f"{img_file_name}.json"
		try:
			with pred_file.open('r') as f:
				predictions = json.load(f)

			for pred in predictions:
				pred['mask'] = mask_conversions.rle_to_bin_mask(pred['mask'])

			# Could test keys and values too, but whatever
		except FileNotFoundError:
			return []

		return predictions

	def get_n_images_with_predictions(self, model_name: str) -> int:
		n_images_with_preds = 0
		for img_file_name in self._img_files_for_model(model_name):
			predictions = self.load(img_file_name, model_name)
			if len(predictions) >= 1:
				n_images_with_preds += 1
		return n_images_with_preds

	def get_n_objects(self, model_name: str) -> int:
		return sum(self.class_distribution(model_name).values())
	
	def class_distribution(self, model_name: str) -> dict[str, int]:
		if hasattr(self, '_computed_class_dists') and model_name in self._computed_class_dists:
			return self._computed_class_dists[model_name]
		
		class_dist = defaultdict(lambda: 0)
		for img_file_name in self._img_files_for_model(model_name):
			predictions = self.load(img_file_name, model_name)
			for pred in predictions:
				class_dist[pred['classname']] += 1
		class_dist = dict(class_dist)

		if not hasattr(self, '_computed_class_dists'):
			self._computed_class_dists = {}
		self._computed_class_dists[model_name] = class_dist

		return class_dist

	def get_classnames(self, model_name: str) -> list[str]:
		return self.class_distribution(model_name).keys()

	def filter(self, out_dir: Path, model_name: str, classes: list[str] = None, img_file_name: str = None):
		if classes is None and img_file_name is None:
			raise ValueError("Can't filter without specifying either classes or img_file_name")
		if classes is not None and img_file_name is not None:
			raise ValueError("Filtering by both classes and img at once is not supported")

		actual_out_dir = out_dir / model_name
		actual_out_dir.mkdir(parents=True, exist_ok=True)

		if classes is not None: # filter by classes
			for img_file_name in self._img_files_for_model(model_name):
				predictions = self.load(img_file_name, model_name)

				filtered_preds = []
				for pred in predictions:
					if pred['classname'] in classes:
						filtered_preds.append(pred)
			
				for pred in filtered_preds:
					pred['mask'] = mask_conversions.bin_mask_to_rle(pred['mask'])

				with (actual_out_dir / f"{img_file_name}.json").open('w') as f:
					json.dump(filtered_preds, f)
		else: # filter by img
			filtered_preds = self.load(img_file_name, model_name)

			for pred in filtered_preds:
				pred['mask'] = mask_conversions.bin_mask_to_rle(pred['mask'])
			
			with (actual_out_dir / f"{img_file_name}.json").open('w') as f:
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
					# this format to evaluate)
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

	def _img_files_for_model(self, model_name: str) -> Generator:
		return (f.stem for f in (self.root_dir / model_name).glob('*'))

	@classmethod
	def from_coco_format(cls, coco_pred, classmap) -> dict:
		cat_id = coco_pred['category_id']
		confidence = coco_pred['score']
		bbox = coco_pred['bbox'].tolist()
		rle_mask = coco_pred['segmentation']

		custom_pred = {
			'classname': classmap[cat_id],
			'confidence': confidence,
			'mask': rle_mask,
			'bbox': bbox,
		}
		return custom_pred