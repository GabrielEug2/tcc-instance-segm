"""Functions to work with annotations in COCO format.
(https://cocodataset.org/#format-data)"""

import json
from pathlib import Path

class Annotations:
	def __init__(self, ann_file: Path):
		try:
			anns = self._load_anns(ann_file)
		except Exception:
			raise

		self.annotations = anns['annotations']
		self.img_info = anns['images']
		self.classmap = self._extract_classmap(anns['categories'])

	@classmethod
	def _load_anns(cls, ann_file):
		if not ann_file.exists():
			raise FileNotFoundError(str(ann_file))

		with ann_file.open('r') as f:
			anns = json.load(f)

		# Could test keys and values too, but whatever

		return anns
		
	@staticmethod
	def _extract_classmap(categories):
		classmap = {}
		for cat in categories:
			classmap[str(cat['id'])] = cat['name']
		return classmap

	def normalize_classmap(self):
		sorted_cats = sorted(self.classmap.items(), key=lambda c: int(c[0]))
		n_classes = len(self.classmap)

		normalized_map = {}
		for i in range(n_classes):
			normalized_map[str(i)] = sorted_cats[i][1]

		return normalized_map

	def class_distribution(self):
		class_dist = {}
		for ann in self.annotations:
			cat_id = str(ann['category_id'])
			cat_name = self.classmap[cat_id]
			class_dist[cat_name] = class_dist.get(cat_name, 0) + 1

		return class_dist

	def filter_by_img(self, img_file: Path):
		img_desc = self._get_img_desc(img_file)
		if img_desc == None:
			return []
		
		relevant_anns = self._filter_by_img_id(img_desc['id'])

		return relevant_anns

	def _get_img_desc(self, img_file):
		requested_img_desc = None
		for img_desc in self.img_info:
			if img_desc['file_name'] == img_file.name:
				requested_img_desc = img_desc
				break
		return requested_img_desc

	def _filter_by_img_id(self, img_id):
		relevant_anns = []
		for ann in self.annotations:
			if ann['image_id'] == img_id:
				relevant_anns.append(ann)
		return relevant_anns

	def get_img_dimensions(self, img_file: Path):
		img_desc = self._get_img_desc(img_file)
		if img_desc == None:
			return (0, 0)

		return (img_desc['height'], img_desc['width'])