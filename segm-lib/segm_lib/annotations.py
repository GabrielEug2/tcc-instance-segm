from collections import defaultdict
import json
from pathlib import Path

from segm_lib import mask_conversions

class Annotations:
	"""Functions to work with annotations in my custom format.
	It's basically the same as the predictions format, but
	without confidence (since, for annotations, that's suposedly
	100%)."""

	def __init__(self, ann_dir: Path):
		if not ann_dir.exists():
			raise FileNotFoundError(f"Dir not found {str(ann_dir)}")

		self.root_dir = ann_dir

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

	def get_n_images(self):
		return len(list(self.root_dir.glob('*.json')))

	def class_distribution(self):
		ann_files = self.root_dir.glob('*.json')

		class_dist = defaultdict(lambda: 0)
		for ann_file in ann_files:
			annotations = self._load_from_file(ann_file)

			for ann in annotations:
				class_dist[ann['classname']] += 1

		self._class_dist = class_dist
		return class_dist

	def get_n_objects(self):
		if hasattr(self, '_class_dist'):
			n_objects = 0
			for count in self._class_dist.values():
				n_objects += count
			return count

		count = 0
		ann_files = self.root_dir.glob('*.json')
		for ann_file in ann_files:
			annotations = self._load_from_file(ann_file)

			for ann in annotations:
				count += 1
		return count

	def copy_to(self, out_dir: Path, classes: list[str]):
		# TODO
		for file in self.files:
			shutil.copy(file, out_dir / file.name)

	def to_coco_format(self, out_file: Path):
		# TODO
		with filtered_anns_file.open('w') as f:
			json.dump(filtered_anns, f)