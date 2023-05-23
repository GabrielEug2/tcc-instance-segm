import json
from collections import defaultdict
from pathlib import Path
from typing import Generator

from .. import mask_conversions
from ..structures import Prediction
from ..classname_normalization import normalize_classname

class PredManager:
	"""Functions to work with predictions from in segm_lib format."""

	def __init__(self, pred_dir: Path):
		if not pred_dir.exists():
			pred_dir.mkdir(parents=True)

		self.root_dir = pred_dir

	def save(self, predictions: list[Prediction], img_file_name: str, model_name: str):
		"""Save the predictions for a given model on a given image.

		Args:
			preds (list): list of predictions for a given image.
			img_file_name (str): image the predictions refer to.
			model_name (str): name of the model used to make the
				predictions.
		"""
		serializable_preds = []
		for pred in predictions:
			serializable_preds.append(pred.serializable())

		out_file = self.root_dir / model_name / f'{img_file_name}.json'
		out_file.parent.mkdir(parents=True, exist_ok=True)
		with out_file.open('w') as f:
			json.dump(serializable_preds, f, indent=4)

	def load(self, img_file_name: str, model_name: str) -> list[Prediction]:
		pred_file = self.root_dir / model_name / f'{img_file_name}.json'
		try:
			with pred_file.open('r') as f:
				serializable_preds = json.load(f)
		except FileNotFoundError:
			return []

		predictions = []
		for seri_pred in serializable_preds:
			predictions.append(Prediction.from_serializable(seri_pred))

		return predictions

	def get_model_names(self) -> list[str]:
		return [f.stem for f in self.root_dir.glob('*') if f.is_dir()]		

	def class_distribution(self, model_name: str) -> dict[str, int]:
		if hasattr(self, '_cached_class_dists') and model_name in self._cached_class_dists:
			return self._cached_class_dists[model_name]
		
		class_dist = defaultdict(lambda: 0)
		for img_file_name in self._img_files_for_model(model_name):
			predictions = self.load(img_file_name, model_name)
			for pred in predictions:
				class_dist[pred.classname] += 1
		class_dist = dict(class_dist)

		if not hasattr(self, '_cached_class_dists'):
			self._cached_class_dists = {}
		self._cached_class_dists[model_name] = class_dist

		return class_dist

	def get_n_images_with_predictions(self, model_name: str) -> int:
		n_images_with_preds = 0

		for img_file_name in self._img_files_for_model(model_name):
			predictions = self.load(img_file_name, model_name)
			if len(predictions) >= 1:
				n_images_with_preds += 1

		return n_images_with_preds

	def get_n_objects(self, model_name: str) -> int:
		return sum(self.class_distribution(model_name).values())

	def filter(self, out_dir: Path, model_name: str, classes: list[str] = None, img_file_name: str = None):
		"""Filter the predictions by the specified criteria.

		Args:
			out_dir (Path): file to write the filtered annotations.
			model_name (str): model to filter for.
			classes (list[str], optional): classes to keep.
			img_file_name (str, optional): image to filter for.

		Raises:
			ValueError: if invalid filtering params were given.
		"""
		if classes is None and img_file_name is None:
			raise ValueError('Cannot filter without specifying either classes or img_file_name')
		if classes is not None and img_file_name is not None:
			raise ValueError('Filtering by both classes and img_file_name at once is not supported')

		filtered_pred_manager = PredManager(out_dir)
		if classes is not None:
			self._filter_by_classes(model_name, classes, filtered_pred_manager)
		else:
			self._filter_by_img(model_name, img_file_name, filtered_pred_manager)

	def normalize_classnames(self):
		for model in self.get_model_names():
			for img in self._img_files_for_model(model):
				predictions = self.load(img, model)

				for pred in predictions:
					pred.classname = normalize_classname(pred.classname)

				self.save(predictions, img, model)

	def to_coco_format(self, model_name: str, img_map: dict, classmap: dict, out_file: Path):
		from .coco_pred_manager import COCOPredManager

		coco_preds = []
		for img_file_name in img_map:
			predictions = self.load(img_file_name, model_name)

			for pred in predictions:
				classname = pred.classname
				if classname not in classmap:
					# would have to give a new id, but since there's no annotations
					# it doesn't make sense to keep them in my case (I only use
					# this format to evaluate)
					continue

				formatted_pred = {
					"image_id": img_map[img_file_name],
					"category_id": classmap[classname],
					"segmentation": mask_conversions.bin_mask_to_rle(pred.mask),
					"score": pred.confidence
				}
				coco_preds.append(formatted_pred)

		coco_pred_manager = COCOPredManager(out_file)
		coco_pred_manager.predictions = coco_preds
		coco_pred_manager.save()

	def _filter_by_classes(self, model_name: str, classes: list[str], filtered_pred_manager: 'PredManager'):
		for img_file_name in self._img_files_for_model(model_name):
			predictions_for_img = self.load(img_file_name, model_name)

			filtered_preds = []
			for pred in predictions_for_img:
				if pred.classname in classes:
					filtered_preds.append(pred)

			filtered_pred_manager.save(filtered_preds, img_file_name, model_name)

	def _filter_by_img(self, model_name: str, img_file_name: str, filtered_pred_manager: 'PredManager'):
		filtered_preds = self.load(img_file_name, model_name)

		filtered_pred_manager.save(filtered_preds, img_file_name, model_name)

	def _img_files_for_model(self, model_name: str) -> Generator:
		return (f.stem for f in (self.root_dir / model_name).glob('*'))