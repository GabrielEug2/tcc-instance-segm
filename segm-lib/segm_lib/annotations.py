from collections import defaultdict
import json
from pathlib import Path
from typing import Generator

from segm_lib import mask_conversions

class Annotations:
	"""Functions to work with annotations in my segm_lib format.
	It's basically the same as the predictions format, but
	without confidence (since, for annotations, that's suposedly
	100%)."""

	def __init__(self, ann_dir: Path):
		if not ann_dir.exists():
			raise FileNotFoundError(f"Dir not found {str(ann_dir)}")

		self.root_dir = ann_dir

	def get_img_file_names(self):
		return (f.stem for f in self.root_dir.glob('*.json'))

	def load(self, img_file_name: str) -> dict:
		ann_file = self.root_dir / f"{img_file_name}.json"
		try:
			with ann_file.open('r') as f:
				annotations = json.load(f)

			for ann in annotations:
				ann['mask'] = mask_conversions.rle_to_bin_mask(ann['mask'])

			# Could test keys and values too, but whatever
		except Exception:
			annotations = []

		return annotations

	def get_n_images(self) -> int:
		# 1 image per file, so...
		n_images = sum(1 for _ in self._all_files())
		return n_images

	def get_n_objects(self) -> int:
		return sum(self.class_distribution().values())

	def class_distribution(self) -> dict[str, int]:
		if hasattr(self, '_class_dist'):
			return self._class_dist
		
		class_dist = defaultdict(lambda: 0)
		for img_file_name in self.get_img_file_names():
			annotations = self.load(img_file_name)
			for ann in annotations:
				class_dist[ann['classname']] += 1
		class_dist = dict(class_dist)

		self._class_dist = class_dist
	
		return class_dist

	def get_classnames(self) -> list[str]:
		return self.class_distribution().keys()
	
	def class_dist_on_img(self, img_file_name: str) -> dict[str, int]:
		annotations = self.load(img_file_name)

		img_class_dist = defaultdict(lambda: 0)
		for ann in annotations:
			img_class_dist[ann['classname']] += 1
		img_class_dist = dict(img_class_dist)

		return img_class_dist

	def filter(self, out_dir: Path, classes: list[str] = None, img_file_name: str = None):
		if classes is None and img_file_name is None:
			raise ValueError("Can't filter without specifying either classes or img_file_name")
		if classes is not None and img_file_name is not None:
			raise ValueError("Filtering by both classes and img at once is not supported")

		out_dir.mkdir(parents=True, exist_ok=True)

		if classes is not None: # filter by classes
			for img_file_name in self.get_img_file_names():
				annotations = self.load(img_file_name)

				filtered_anns = []
				for ann in annotations:
					if ann['classname'] in classes:
						filtered_anns.append(ann)
			
				for ann in filtered_anns:
					ann['mask'] = mask_conversions.bin_mask_to_rle(ann['mask'])

				with (out_dir / f"{img_file_name}.json").open('w') as f:
					json.dump(filtered_anns, f)
		else: # filter by img
			filtered_anns = self.load(img_file_name)

			for ann in filtered_anns:
				ann['mask'] = mask_conversions.bin_mask_to_rle(ann['mask'])
			
			with (out_dir / f"{img_file_name}.json").open('w') as f:
				json.dump(filtered_anns, f)

	def _all_files(self):
		return self.root_dir.glob('*.json')
	
	@classmethod
	def from_coco_format(cls, preds, classmap) -> dict:
		# TODO implement
		return