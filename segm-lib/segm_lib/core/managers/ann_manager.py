import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Generator

from .. import mask_conversions
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

		for img_file_name in coco_anns.img_file_names():
			tmp_file = Path('tmp', f'{img_file_name}_coco-anns.json')
			coco_anns.filter(tmp_file, img_file_name=img_file_name)
			coco_anns_for_img = COCOAnnManager(tmp_file)
			
			img_h, img_w = img_dimensions[img_file_name]

			custom_anns = []
			for coco_ann in coco_anns_for_img.annotations:
				cat_id = coco_ann['category_id']
				classname = classmap_by_id[cat_id]

				rle = mask_conversions.ann_to_rle(coco_ann['segmentation'], img_h, img_w)
				mask = mask_conversions.rle_to_bin_mask(rle)

				bbox = coco_ann['bbox'] # Mesmo formato do COCO, [x, y, w, h]

				custom_anns.append(Annotation(classname, mask, bbox))

			self._save(custom_anns, img_file_name)
			os.remove(str(tmp_file))

	def get_n_images(self) -> int:
		# 1 image per file, so...
		n_images = sum(1 for _ in self.root_dir.glob('*.json'))
		return n_images

	def get_n_objects(self) -> int:
		return sum(self.class_distribution().values())

	def class_distribution(self) -> dict[str, int]:
		if hasattr(self, '_cached_class_dist'):
			return self._cached_class_dist
		
		class_dist = defaultdict(lambda: 0)
		for img_file_name in self.get_img_file_names():
			annotations = self.load(img_file_name)
			for ann in annotations:
				class_dist[ann.classname] += 1
		class_dist = dict(class_dist)

		self._cached_class_dist = class_dist
		return class_dist

	def get_img_file_names(self) -> Generator[str, None, None]:
		return (f.stem for f in self.root_dir.glob('*.json'))

	def class_dist_on_img(self, img_file_name: str) -> dict[str, int]:
		annotations = self.load(img_file_name)

		class_dist = defaultdict(lambda: 0)
		for ann in annotations:
			class_dist[ann.classname] += 1
		class_dist = dict(class_dist)

		return class_dist

	def load(self, img_file_name: str) -> list[Annotation]:
		ann_file = self.root_dir / f"{img_file_name}.json"
		try:
			with ann_file.open('r') as f:
				serializable_anns = json.load(f)
		except FileNotFoundError:
			return []

		annotations = []
		for seri_ann in serializable_anns:
			classname = seri_ann['classname']
			mask = mask_conversions.rle_to_bin_mask(seri_ann['mask'])
			bbox = seri_ann['bbox']

			pred_obj = Annotation(classname, mask, bbox)

			annotations.append(pred_obj)

		return annotations
	
	def normalize_classnames(self):
		for img in self.get_img_file_names():
			annotations = self.load(img)

			for ann in annotations:
				ann.classname = ann.classname.lower()

			self._save(annotations, img)
	
	def get_classnames(self) -> set[str]:
		return self.class_distribution().keys()
	
	def filter(self, out_dir: Path, classes: list[str] = None, img_file_name: str = None):
		"""Filter the annotations by the specified criteria.

		Args:
			out_dir (Path): file to write the filtered annotations.
			classes (list[str], optional): classes to keep.
			img_file_name (str, optional): image to filter for.

		Raises:
			ValueError: if invalid filtering params were given.
		"""
		if classes is None and img_file_name is None:
			raise ValueError('Cannot filter without specifying either classes or img_file_name')
		if classes is not None and img_file_name is not None:
			raise ValueError('Filtering by both classes and img_file_name at once is not supported')

		filtered_ann_manager = AnnManager(out_dir)
		if classes is not None:
			self._filter_by_classes(classes, filtered_ann_manager)
		else:
			self._filter_by_img(img_file_name, filtered_ann_manager)

	def _filter_by_classes(self, classes: list[str], filtered_ann_manager: 'AnnManager'):
		for img_file_name in self.get_img_file_names():
			annotations_for_img = self.load(img_file_name)

			filtered_anns = [a for a in annotations_for_img if a.classname in classes]
		
			filtered_ann_manager._save(filtered_anns, img_file_name)

	def _filter_by_img(self, img_file_name: str, filtered_ann_manager: 'AnnManager'):
		filtered_anns = self.load(img_file_name)

		filtered_ann_manager._save(filtered_anns, img_file_name)

	def _save(self, anns: list[Annotation], img_file_name: str):
		serializable_anns = []
		for ann in anns:
			ann_dict = {}
			ann_dict['classname'] = ann.classname
			ann_dict['mask'] = mask_conversions.bin_mask_to_rle(ann.mask)
			ann_dict['bbox'] = ann.bbox

			serializable_anns.append(ann_dict)

		out_file = self.root_dir / f'{img_file_name}.json'
		with out_file.open('w') as f:
			json.dump(serializable_anns, f, indent=4)