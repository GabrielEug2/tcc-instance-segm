import json
from collections import defaultdict
from pathlib import Path

from ..classname_normalization import normalize

class COCOAnnManager:
	"""Functions to work with annotations in COCO format.
	(https://cocodataset.org/#format-data)"""

	def __init__(self, ann_file: Path):
		if not ann_file.exists():
			ann_file.parent.mkdir(parents=True, exist_ok=True)

			self.file = ann_file
			self.images = []
			self.categories = []
			self.annotations = []
			self._save()
			return

		with ann_file.open('r') as f:
			anns = json.load(f)
		# Could test keys and values too, but whatever
		
		self.file = ann_file
		self.images = anns['images']
		self.categories = anns['categories']
		self.annotations = anns['annotations']

	def classmap(self) -> dict[str, int]:
		if hasattr(self, '_cached_classmap'):
			return self._cached_classmap

		classmap = {}
		for cat in self.categories:
			classmap[cat['name']] = cat['id']

		self._cached_classmap = classmap
		return classmap

	def normalized_classmap(self) -> dict[str, int]:
		raw_map = self.classmap()

		classmap_sorted_by_id = sorted(raw_map.items(), key=lambda c: int(c[1]))

		normalized_map = {}
		for i, (classname, old_id) in enumerate(classmap_sorted_by_id):
			normalized_map[classname] = i

		return normalized_map

	def classmap_by_id(self) -> dict[int, str]:
		return {id_: name for name, id_ in self.classmap().items()}
	
	def class_distribution(self) -> dict[str, int]:
		classmap_by_id = self.classmap_by_id()

		class_dist = defaultdict(lambda: 0)
		for ann in self.annotations:
			cat_id = ann['category_id']
			cat_name = classmap_by_id[cat_id]
			class_dist[cat_name] += 1
		class_dist = dict(class_dist)

		return class_dist

	def img_file_names(self) -> list[str]:
		return self.img_map().keys()

	def img_map(self) -> dict[str, int]:
		if hasattr(self, '_cached_img_map'):
			return self._cached_img_map
	
		img_map = {}
		for image in self.images:
			# tanto faz a extensão, eu só considero o nome
			img_map[Path(image['file_name']).stem] = image['id']

		self._cached_img_map = img_map
		return img_map
	
	def img_dimensions(self) -> dict[str, tuple[int, int]]:
		img_dimensions = {}

		for image in self.images:
			img_dimensions[Path(image['file_name']).stem] = (image['height'], image['width'])

		return img_dimensions

	def normalize_classnames(self):
		for cat in self.categories:			
			cat['name'] = normalize(cat['name'])
		
		self._save()

	def filter(self, out_file: Path, classes: list[str] = None, img_file_name: str = None):
		"""Filter the annotations by the specified criteria.

		Args:
			out_file (Path): file to write the filtered annotations.
			classes (list[str], optional): classes to keep.
			img_file_name (str, optional): image to filter for.

		Raises:
			ValueError: if invalid filtering params were given.
		"""
		if classes is None and img_file_name is None:
			raise ValueError('Cannot filter without specifying either classes or img_file_name')
		if classes is not None and img_file_name is not None:
			raise ValueError('Filtering by both classes and img at once is not supported')

		filtered_anns = COCOAnnManager(out_file)
		if classes is not None:
			self._filter_by_classes(classes, filtered_anns)
		else:
			self._filter_by_img(img_file_name, filtered_anns)

	def _filter_by_classes(self, classes: list[str], filtered_anns: 'COCOAnnManager'):
		cat_ids_to_keep = []
		for cat_name, cat_id in self.classmap().items():
			if cat_name in classes:
				cat_ids_to_keep.append(cat_id)

		filtered_anns.images = self.images
		filtered_anns.categories = [c for c in self.categories if c['id'] in cat_ids_to_keep]
		filtered_anns.annotations = [a for a in self.annotations if a['category_id'] in cat_ids_to_keep]

		filtered_anns._save()

	def _filter_by_img(self, img_file_name: str, filtered_anns: 'COCOAnnManager'):
		img_map = self.img_map()
		img_id = img_map[img_file_name]

		img_desc = next((i for i in self.images if i['id'] == img_id), None)
		if img_desc is None:
			raise ValueError(f'Info for img {img_file_name} not found in annotations')
		filtered_anns.images = [img_desc]

		filtered_anns.annotations = [a for a in self.annotations if a['image_id'] == img_id]

		cat_ids_to_keep = {a['category_id'] for a in filtered_anns.annotations}
		filtered_anns.categories = [c for c in self.categories if c['id'] in cat_ids_to_keep]

		filtered_anns._save()

	def _save(self):
		anns_dict = {}
		anns_dict['images'] = self.images
		anns_dict['categories'] = self.categories
		anns_dict['annotations'] = self.annotations

		with self.file.open('w') as f:
			json.dump(anns_dict, f)