"""Functions to work with annotations in my custom format.
It's basically the same as the predictions format."""

import json
from pathlib import Path

class Annotations:
	def __init__(self, ann_file: Path|None = None):
		if ann_file is None:
			self.anns = None
			return

		try:
			anns = self._load_from_file(ann_file)
		except Exception:
			raise
		self.anns = anns

	@classmethod
	def _load_from_file(cls, ann_file):
		if not ann_file.exists():
			raise FileNotFoundError(str(ann_file))

		with ann_file.open('r') as f:
			anns = json.load(f)

		# Could test keys and values too, but whatever

		return anns

	def class_distribution(self):
		classmap_by_id = self.classmap_by_id()

		class_dist = {}
		for ann in self.anns['annotations']:
			cat_id = ann['category_id']
			cat_name = classmap_by_id[str(cat_id)]
			class_dist[cat_name] = class_dist.get(cat_name, 0) + 1

		return class_dist

	def filter_by_classes(self, classes):
		classmap_by_id = self.classmap_by_id()

		relevant_anns = []
		for ann in self.anns['annotations']:
			cat_id = ann['category_id']
			cat_name = classmap_by_id[str(cat_id)]
			if cat_name in classes:
				relevant_anns.append(ann)

		relevant_cats = []
		for cat in self.anns['categories']:
			if cat['name'] in classes:
				relevant_cats.append(cat)

		return {
			'images': self.anns['images'],
			'categories': relevant_cats,
			'annotations': relevant_anns,
		}
	
	def save(self, out_file):
		with out_file.open('w') as f:
			json.dump(self.anns, f)