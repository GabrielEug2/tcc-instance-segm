import json
from collections import defaultdict
from pathlib import Path
from typing import Generator

from segm_lib.core.structures import Prediction


class SingleModelPredManager:
	"""Functions to work with predictions from in segm_lib format."""

	def __init__(self, model_dir: Path):
		if not model_dir.exists():
			model_dir.mkdir(parents=True)

		self.model_dir = model_dir

	def save(self, predictions: list[Prediction], img_name: str):
		"""Save the predictions for a given image.

		Args:
			preds (list): list of predictions for a given image.
			img_name (str): image the predictions refer to.
		"""
		serializable_preds = []
		for pred in predictions:
			serializable_preds.append(pred.serializable())

		out_file = self.model_dir / f'{img_name}.json'
		with out_file.open('w') as f:
			json.dump(serializable_preds, f, indent=4)

	def load(self, img_name: str) -> list[Prediction]:
		pred_file = self.model_dir / f'{img_name}.json'
		try:
			with pred_file.open('r') as f:
				serializable_preds = json.load(f)
		except FileNotFoundError:
			return []

		predictions = []
		for seri_pred in serializable_preds:
			predictions.append(Prediction.from_serializable(seri_pred))

		return predictions

	def get_n_images_with_predictions(self) -> int:
		n_images_with_preds = 0

		for img_name in self.get_img_names():
			predictions = self.load(img_name)
			if len(predictions) >= 1:
				n_images_with_preds += 1

		return n_images_with_preds

	def get_img_names(self) -> Generator:
		return (f.stem for f in self.model_dir.glob('*'))
	
	def get_n_objects(self, img_name: str = None) -> int:
		if img_name is None:
			return sum(self.class_distribution().values())
		else:
			return sum(self.class_distribution(img_name=img_name).values())

	def class_distribution(self, img_name: str = None) -> dict[str, int]:
		if img_name is None and hasattr(self, '_cached_class_dist'):
			return self._cached_class_dist

		class_dist = None
		if img_name is None:
			class_dist = self._class_dist_on_all_imgs()

			if not hasattr(self, '_cached_class_dist'):
				self._cached_class_dist = class_dist
		else:
			class_dist = self._class_dist_on_img(img_name)
		
		return class_dist

	def filter(self, out_dir: Path, classes: list[str] = None, img_name: str = None):
		"""Filter the predictions by the specified criteria.

		Args:
			out_dir (Path): file to write the filtered annotations.
			classes (list[str], optional): classes to keep.
			img_name (str, optional): image to filter for.

		Raises:
			ValueError: if invalid filtering params were given.
		"""
		if classes is None and img_name is None:
			raise ValueError('Cannot filter without specifying either classes or img_name')
		if classes is not None and img_name is not None:
			raise ValueError('Filtering by both classes and img_name at once is not supported')

		filtered_pred_manager = self.__class__(out_dir)
		if classes is not None:
			self._filter_by_classes(classes, filtered_pred_manager)
		else:
			self._filter_by_img(img_name, filtered_pred_manager)

	def to_coco_format(self, img_map: dict, classmap: dict, out_file: Path):
		from .coco_pred_manager import COCOPredManager

		coco_preds = []
		for img_name in img_map:
			predictions = self.load(img_name)

			for pred in predictions:
				classname = pred.classname
				if classname not in classmap:
					# would have to give a new id, but since there's no annotations
					# it doesn't make sense to keep them in my case (I only use
					# the coco format to evaluate)
					continue

				formatted_pred = {
					"image_id": img_map[img_name],
					"category_id": classmap[classname],
					"segmentation": pred.mask,
					"score": pred.confidence
				}
				coco_preds.append(formatted_pred)

		coco_pred_manager = COCOPredManager(out_file)
		coco_pred_manager.predictions = coco_preds
		coco_pred_manager.save()

	def _class_dist_on_all_imgs(self) -> dict[str, int]:
		class_dist = defaultdict(lambda: 0)
		for img_name in self.get_img_names():
			class_dist_on_img = self._class_dist_on_img(img_name)

			for k, v in class_dist_on_img.items():
				class_dist[k] += v

		return dict(class_dist)

	def _class_dist_on_img(self, img_name: str) -> dict[str, int]:
		class_dist = defaultdict(lambda: 0)
		predictions = self.load(img_name)

		for pred in predictions:
			class_dist[pred.classname] += 1

		return dict(class_dist)

	def _filter_by_classes(self, classes: list[str], filtered_pred_manager: 'SingleModelPredManager'):
		for img_name in self.get_img_names():
			predictions_for_img = self.load(img_name)

			filtered_preds = [p for p in predictions_for_img if p.classname in classes]

			filtered_pred_manager.save(filtered_preds, img_name)

	def _filter_by_img(self, img_name: str, filtered_pred_manager: 'SingleModelPredManager'):
		filtered_preds = self.load(img_name)

		filtered_pred_manager.save(filtered_preds, img_name)