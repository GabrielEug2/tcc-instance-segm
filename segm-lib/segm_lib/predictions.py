from collections import defaultdict
import json
from pathlib import Path
import shutil
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
			n_images += 1
		return n_images

	def _pred_files_for_model(self, model_name: str) -> Generator:
		return (self.root_dir / model_name).glob('*')

	def get_n_objects(self, model_name: str) -> int:
		if hasattr(self, '_class_dist'):
			n_objects = 0
			for count in self._class_dist.values():
				n_objects += count
			return count

		count = 0
		for pred_file in self._pred_files_for_model(model_name):
			predictions = self._load_from_file(pred_file)

			for pred in predictions:
				count += 1
		return count
	
	def class_distribution(self, model_name: str) -> dict:
		class_dist = defaultdict(lambda: 0)
		for pred_file in self._pred_files_for_model(model_name):
			predictions = self._load_from_file(pred_file)

			for pred in predictions:
				class_dist[pred['classname']] += 1

		self._class_dist = class_dist
		return class_dist

	def copy_to(self, out_dir: Path, model_name: str, classes: list[str]):
		for file in self.files:
			shutil.copy(file, out_dir / file.name)

	def to_coco_format(self, out_file: Path, model_name: str):
		# TODO
		coco_formatted_data = []
		for img_file_name in img_map:
			predictions = pred_manager.load(img_file_name, model_name)

			for pred in predictions:
				classname = pred['classname']
				if classname not in evaluatable_classes:
					# can't evaluate if there are no annotations
					# of that class to compare
					continue

				mask = mask_conversions.bin_mask_to_rle(pred['mask'])

				formatted_pred = {
					"image_id": img_map[img_file_name],
					"category_id": classmap[classname],
					"segmentation": mask,
					"score": pred['confidence']
				}
				coco_formatted_data.append(formatted_pred)

		coco_formatted_pred_file = out_dir / 'predictions_used_for_eval.json'
		coco_formatted_pred_file.parent.mkdir(parents=True, exist_ok=True)
		with coco_formatted_pred_file.open('w') as f:
			json.dump(coco_formatted_data, f)