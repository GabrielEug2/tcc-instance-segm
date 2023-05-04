"""Functions to work with annotations in COCO format.
(https://cocodataset.org/#format-data)"""

import json
from pathlib import Path

class COCO_AnnParser:
	# def __init__(self, ann_file: Path|None = None):
	# 	if ann_file is None:
	# 		self.anns = None
	# 		return
	def __init__(self, ann_file: Path):
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

	def classmap(self):
		if self._classmap:
			return self._classmap

		classmap = {}
		for cat in self.anns['categories']:
			classmap[cat['name']] = cat['id']

		self._classmap = classmap
		return classmap

	def normalized_classmap(self):
		raw_map = self.classmap()

		map_sorted_by_id = sorted(raw_map.items(), key=lambda c: int(c[1]))
		n_classes = len(raw_map)

		normalized_map = {}
		for i in range(n_classes):
			normalized_map[map_sorted_by_id[i][0]] = i

		return normalized_map

	def classmap_by_id(self):
		return {str(id_): name for name, id_ in self.classmap().items()}
	
	def class_distribution(self):
		classmap_by_id = self.classmap_by_id()

		class_dist = {}
		for ann in self.anns['annotations']:
			cat_id = ann['category_id']
			cat_name = classmap_by_id[str(cat_id)]
			class_dist[cat_name] = class_dist.get(cat_name, 0) + 1

		return class_dist

	def img_map(self):
		img_map = {}
		for image in self.anns['images']:
			# tanto faz a extensão, eu só considero o nome
			img_map[Path(image['file_name']).stem] = image['id']
		return img_map

	def filter_by_img_id(self, img_id: int):
		relevant_anns = []
		for ann in self.anns['annotations']:
			if ann['image_id'] == img_id:
				relevant_anns.append(ann)
		return relevant_anns

	# @classmethod
	# def from_dict(cls, anns):
	# 	# Could test keys and values too, but whatever
	# 	anns_obj = cls()
	# 	anns_obj.anns = anns
	# 	return anns_obj
	
	# def img_dimensions(self):
	# 	img_dimensions = {}
	# 	for image in self.anns['images']:
	# 		img_dimensions[Path(image['file_name']).stem] = (image['height'], image['width'])
	# 	return img_dimensions

	# def filter_by_img(self, img_file_name: str):
	# 	img_map = self.img_map()
	# 	img_id = img_map[img_file_name]
	# 	relevant_anns = self._filter_by_img_id(img_id)
	# 	return relevant_anns


	
	# def normalize_classnames(self):
	# 	for cat in self.anns['categories']:
	# 		cat['name'] = cat['name'].lower()

	# def filter_by_classes(self, classes):
	# 	classmap_by_id = self.classmap_by_id()

	# 	relevant_anns = []
	# 	for ann in self.anns['annotations']:
	# 		cat_id = ann['category_id']
	# 		cat_name = classmap_by_id[str(cat_id)]
	# 		if cat_name in classes:
	# 			relevant_anns.append(ann)

	# 	relevant_cats = []
	# 	for cat in self.anns['categories']:
	# 		if cat['name'] in classes:
	# 			relevant_cats.append(cat)

	# 	return {
	# 		'images': self.anns['images'],
	# 		'categories': relevant_cats,
	# 		'annotations': relevant_anns,
	# 	}
	
	# def save(self, out_file):
	# 	with out_file.open('w') as f:
	# 		json.dump(self.anns, f)