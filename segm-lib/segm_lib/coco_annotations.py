from collections import defaultdict
import json
from pathlib import Path

class COCOAnnotations:
	"""Functions to work with annotations in COCO format.
	(https://cocodataset.org/#format-data)"""

	def __init__(self, ann_file: Path):
		if not ann_file.exists():
			raise FileNotFoundError(str(ann_file))

		with ann_file.open('r') as f:
			anns = json.load(f)

		# Could test keys and values too, but whatever

		self._anns = anns

	def classmap(self) -> dict[str, int]:
		if hasattr(self, '_classmap'):
			return self._classmap

		classmap = {}
		for cat in self._anns['categories']:
			classmap[cat['name']] = cat['id']

		self._classmap = classmap
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
	
	def class_distribution(self):
		classmap_by_id = self.classmap_by_id()

		class_dist = defaultdict(lambda: 0)
		for ann in self._anns['annotations']:
			cat_id = ann['category_id']
			cat_name = classmap_by_id[cat_id]
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

	def filter(self, out_file: Path, classes: list[str] = None, img_file_name: str = None):
		if classes is None and img_file_name is None:
			raise ValueError("Can't filter without specifying either classes or img_file_name")
		if classes is not None and img_file_name is not None:
			raise ValueError("Filtering by both classes and img at once is not supported")

		if classes is not None: # filter by classes
			cat_ids_to_keep = []
			for cat_name, cat_id in self.classmap().items():
				if cat_name.lower() in classes:
					cat_ids_to_keep.append(cat_id)
			
			self._anns['categories'] = [c for c in self._anns['categories'] if c['id'] in cat_ids_to_keep]
			self._anns['annotations'] = [a for a in self._anns['annotations'] if a['category_id'] in cat_ids_to_keep]
		else: # filter by img
			img_map = self.img_map()
			img_id = img_map[img_file_name]

			img_desc = None
			for img in self._anns['images']:
				if img['id'] == img_id:
					img_desc = img
					break
			self._anns['images'] = [img_desc]

			self._anns['annotations'] = self.filter_by_img_id(img_id)

			cat_ids_to_keep = []
			for ann in self._anns['annotations']:
				if ann['category_id'] not in cat_ids_to_keep:
					cat_ids_to_keep.append(ann['category_id'])
			self._anns['categories'] = [c for c in self._anns['categories'] if c['id'] in cat_ids_to_keep]

		out_file.parent.mkdir(parents=True, exist_ok=True)
		with out_file.open('w') as f:
			json.dump(self._anns, f)