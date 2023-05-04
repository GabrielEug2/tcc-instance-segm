from collections import defaultdict
import json
from pathlib import Path

class COCOAnnParser:
	"""Functions to work with annotations in COCO format.
	(https://cocodataset.org/#format-data)"""

	def __init__(self, ann_file: Path):
		self.file = ann_file

	def _load(self):
		if not self.file.exists():
			raise FileNotFoundError(str(self.file))

		with self.file.open('r') as f:
			anns = json.load(f)

		# Could test keys and values too, but whatever

		self._anns = anns

	def classmap(self):
		if hasattr(self, '_classmap'):
			return self._classmap

		if not hasattr(self, '_anns'):
			self._load()

		classmap = {}
		for cat in self._anns['categories']:
			classmap[cat['name']] = cat['id']

		self._classmap = classmap
		return classmap

	def normalized_classmap(self):
		raw_map = self.classmap()

		classmap_sorted_by_id = sorted(raw_map.items(), key=lambda c: int(c[1]))

		normalized_map = {}
		for i, (classname, old_id) in enumerate(classmap_sorted_by_id):
			normalized_map[classname] = i

		return normalized_map

	def classmap_by_id(self):
		return {str(id_): name for name, id_ in self.classmap().items()}
	
	def class_distribution(self):
		classmap_by_id = self.classmap_by_id()

		class_dist = defaultdict(lambda: 0)
		for ann in self._anns['annotations']:
			cat_id = ann['category_id']
			cat_name = classmap_by_id[str(cat_id)]
			class_dist[cat_name] += 1

		return class_dist

	def img_map(self):
		img_map = {}
		for image in self._anns['images']:
			# tanto faz a extensão, eu só considero o nome
			img_map[Path(image['file_name']).stem] = image['id']
		return img_map

	def filter_by_img_id(self, img_id: int):
		relevant_anns = []
		for ann in self._anns['annotations']:
			if ann['image_id'] == img_id:
				relevant_anns.append(ann)
		return relevant_anns

	def img_dimensions(self):
		img_dimensions = {}
		for image in self._anns['images']:
			img_dimensions[Path(image['file_name']).stem] = (image['height'], image['width'])
		return img_dimensions