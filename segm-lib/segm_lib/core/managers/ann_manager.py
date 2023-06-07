import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Generator

from .. import mask_conversions
from ..classname_normalization import normalize_classname
from ..structures import Annotation


class AnnManager:
	"""Functions to work with annotations in segm_lib format."""

	def __init__(self, ann_dir: Path):
		if not ann_dir.exists():
			ann_dir.mkdir(parents=True)

		self.root_dir = ann_dir

	def from_coco_file(self, coco_file: Path):
		from .coco_ann_manager import COCOAnnManager

		coco_anns = COCOAnnManager(coco_file)
		classmap_by_id = coco_anns.classmap_by_id()
		img_dimensions = coco_anns.img_dimensions()

		for img_name in coco_anns.img_names():
			tmp_file = Path('tmp', f'{img_name}_coco-anns.json')
			coco_anns.filter(tmp_file, img_name=img_name)

			coco_anns_for_img = COCOAnnManager(tmp_file)
			img_h, img_w = img_dimensions[img_name]

			custom_anns = []
			for coco_ann in coco_anns_for_img.annotations:
				cat_id = coco_ann['category_id']
				classname = classmap_by_id[cat_id]

				mask = mask_conversions.ann_to_rle(coco_ann['segmentation'], img_h, img_w)

				bbox = coco_ann['bbox'] # Mesmo formato do COCO, [x, y, w, h]

				custom_anns.append(Annotation(classname, mask, bbox))

			self._save(custom_anns, img_name)
			os.remove(str(tmp_file))

	def load(self, img_name: str) -> list[Annotation]:
		ann_file = self.root_dir / f"{img_name}.json"
		try:
			with ann_file.open('r') as f:
				serializable_anns = json.load(f)
		except FileNotFoundError:
			return []

		annotations = []
		for seri_ann in serializable_anns:
			annotations.append(Annotation.from_serializable(seri_ann))

		return annotations

	def get_n_images(self) -> int:
		n_images = sum(1 for _ in self.get_img_names())
		return n_images

	def get_n_objects(self, img_name: str = None) -> int:
		if img_name is None:
			return sum(self.class_distribution().values())
		else:
			return sum(self.class_distribution(img_name).values())

	def class_distribution(self, img_name: str = None) -> dict[str, int]:
		if img_name is None and hasattr(self, '_cached_class_dist'):
			return self._cached_class_dist
		
		class_dist = None
		if img_name is None:
			class_dist = self._class_dist_on_all_imgs()
			self._cached_class_dist = class_dist
		else:
			class_dist = self._class_dist_on_img(img_name)

		return class_dist

	def _class_dist_on_all_imgs(self) -> dict[str, int]:
		class_dist = defaultdict(lambda: 0)
		for img_name in self.get_img_names():
			class_dist_on_img = self._class_dist_on_img(img_name)

			for k, v in class_dist_on_img.items():
				class_dist[k] += v
		class_dist = dict(class_dist)

		return class_dist
	
	def _class_dist_on_img(self, img_name: str) -> dict[str, int]:
		annotations = self.load(img_name)

		class_dist = defaultdict(lambda: 0)
		for ann in annotations:
			class_dist[ann.classname] += 1
		class_dist = dict(class_dist)

		return class_dist

	def get_classnames(self) -> set[str]:
		return self.class_distribution().keys()
	
	def get_img_names(self) -> Generator[str, None, None]:
		return (f.stem for f in self.root_dir.glob('*.json'))
		
	def filter(self, out_dir: Path, classes: list[str] = None, img_name: str = None):
		"""Filter the annotations by the specified criteria.

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

		filtered_ann_manager = AnnManager(out_dir)
		if classes is not None:
			self._filter_by_classes(classes, filtered_ann_manager)
		else:
			self._filter_by_img(img_name, filtered_ann_manager)
	
	def normalize_classnames(self):
		for img in self.get_img_names():
			annotations = self.load(img)

			for ann in annotations:
				ann.classname = normalize_classname(ann.classname)

			self._save(annotations, img)

	def _filter_by_classes(self, classes: list[str], filtered_ann_manager: 'AnnManager'):
		for img_name in self.get_img_names():
			annotations_for_img = self.load(img_name)

			filtered_anns = [a for a in annotations_for_img if a.classname in classes]
		
			filtered_ann_manager._save(filtered_anns, img_name)

	def _filter_by_img(self, img_name: str, filtered_ann_manager: 'AnnManager'):
		filtered_anns = self.load(img_name)

		filtered_ann_manager._save(filtered_anns, img_name)

	def _save(self, anns: list[Annotation], img_name: str):
		serializable_anns = []
		for ann in anns:
			serializable_anns.append(ann.serializable())

		out_file = self.root_dir / f'{img_name}.json'
		with out_file.open('w') as f:
			json.dump(serializable_anns, f, indent=4)