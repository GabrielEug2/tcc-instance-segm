from collections import defaultdict
import json
from pathlib import Path

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

	def _all_files(self):
		return self.root_dir.glob('*.json')

	def load(self, img_file_name: str) -> dict:
		ann_file = self.root_dir / f"{img_file_name}.json"
		try:
			annotations = self._load_from_file(ann_file)
		except FileNotFoundError:
			annotations = []

		return annotations

	def _load_from_file(self, ann_file):
		with ann_file.open('r') as f:
			annotations = json.load(f)

		for ann in annotations:
			ann['mask'] = mask_conversions.rle_to_bin_mask(ann['mask'])

		# Could test keys and values too, but whatever

		return annotations

	def get_n_images(self) -> int:
		# 1 image per file, so...
		n_images = len(list(self._all_files))
		return n_images

	def class_distribution(self) -> dict[str, int]:
		if hasattr(self, '_class_dist'):
			return self._class_dist
		
		class_dist = defaultdict(lambda: 0)
		for ann_file in self._all_files():
			annotations = self._load_from_file(ann_file)

			for ann in annotations:
				class_dist[ann['classname']] += 1

		self._class_dist = class_dist
		return class_dist

	def get_n_objects(self) -> int:
		if hasattr(self, '_class_dist'):
			class_dist = self._class_dist
		else:
			class_dist = self.class_distribution()

		n_objects = 0
		for count in class_dist.values():
			n_objects += count
		return count

	def get_classnames(self) -> list[str]:
		return self.class_distribution().keys()

	def filter(self, classes: list[str], out_dir: Path):
		out_dir.mkdir(parents=True, exist_ok=True)

		for ann_file in self._all_files():
			annotations = self._load_from_file(ann_file)

			filtered_anns = []
			for ann in annotations:
				if ann['classname'] in classes:
					filtered_anns.append(ann)
			
			with (out_dir / ann_file.name).open('w') as f:
				json.dump(filtered_anns, f)