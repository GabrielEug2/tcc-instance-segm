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
		self.classmap = self._extract_classmap(anns['categories'])
		self.img_map = self._extract_img_map(anns['images'])
		self.img_dimensions = self._extract_img_dimensions(anns['images'])

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
			classmap[cat['name']] = cat['id']
		return classmap

	def classmap_by_id(self):
		return {str(id_): name for name, id_ in self.classmap.items()}

	@staticmethod
	def _extract_img_map(images):
		img_map = {}
		for image in images:
			img_map[image['file_name']] = image['id']
		return img_map

	def img_map_by_id(self):
		return {str(id_): filename for filename, id_ in self.img_map.items()}
	
	@staticmethod
	def _extract_img_dimensions(images):
		img_dimensions = {}
		for image in images:
			img_dimensions[image['file_name']] = (image['height'], image['width'])
		return img_dimensions

	def normalize_classmap(self):
		sorted_cats = sorted(self.classmap.items(), key=lambda c: int(c[1]))
		n_classes = len(self.classmap)

		normalized_map = {}
		for i in range(n_classes):
			normalized_map[sorted_cats[i][0]] = i

		return normalized_map

	def class_distribution(self):
		classmap_by_id = self.classmap_by_id()

		class_dist = {}
		for ann in self.annotations:
			cat_id = ann['category_id']
			cat_name = classmap_by_id[str(cat_id)]
			class_dist[cat_name] = class_dist.get(cat_name, 0) + 1

		return class_dist

	def filter_by_img(self, img_file: Path):
		img_id = self.img_map[img_file]
		relevant_anns = self._filter_by_img_id(img_id)
		return relevant_anns

	def _filter_by_img_id(self, img_id):
		relevant_anns = []
		for ann in self.annotations:
			if ann['image_id'] == img_id:
				relevant_anns.append(ann)
		return relevant_anns

	def get_img_dimensions(self, img_file: Path):
		return self.img_dimensions[img_file]


def class_dist_from_multiple_files(ann_files):
	total_class_dist = {}
	for ann_file in ann_files:
		print(f"Processing {ann_file.name}...") # Só pra saber se é normal a demora, tipo o annotations_train
		try:
			file_dist = Annotations(ann_file).class_distribution()
		except Exception as e:
			raise ValueError(f"File \"{str(ann_file)}\" does not follow the expected format") from e

		for classname in file_dist:
			total_class_dist[classname] = total_class_dist.get(classname, 0) + file_dist[classname]

	return total_class_dist