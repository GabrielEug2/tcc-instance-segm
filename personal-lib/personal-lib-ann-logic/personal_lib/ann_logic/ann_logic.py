"""Functions to parse annotations in COCO format.
	(https://cocodataset.org/#format-data)"""

import json
from pathlib import Path

def load_anns(ann_file: Path):
	if not ann_file.exists():
		raise FileNotFoundError(str(ann_file))

	try:
		with ann_file.open('r') as f:
			anns = json.load(f)
	except json.JSONDecodeError as e:
		raise ValueError(f"File \"{str(ann_file)}\" does not follow the expected format") from e

	return anns

def extract_classmap(anns):
	classmap = {}

	for cat in anns['categories']:
		classmap[str(cat['id'])] = cat['name']

	return classmap

def normalize_classmap(raw_map):
	sorted_cats = sorted(raw_map.items(), key=lambda c: int(c[0]))
	n_classes = len(raw_map.keys())

	normalized_map = {}
	for i in range(n_classes):
		normalized_map[str(i)] = sorted_cats[i][1]

	return normalized_map

def class_distribution(ann_files: Path|list[Path]):
	for f in ann_files:
		if not f.exists():
			raise FileNotFoundError(str(f))
	
	if type(ann_files) == Path:
		ann_files = [ann_files]

	class_dist = _class_dist_from_files(ann_files)

	return class_dist

def _class_dist_from_files(ann_files: list[Path]):
	class_dist = {}

	for ann_file in ann_files:
		print(f"Processing {ann_file.name}...") # Só pra saber se é normal a demora, tipo o annotations_train
		try:
			file_dist = _class_dist_from_file(ann_file)
		except Exception as e:
			raise ValueError(f"File \"{str(ann_file)}\" does not follow the expected format") from e

		_update_counts(class_dist, file_dist)

	return class_dist

def _class_dist_from_file(ann_file):
	anns = load_anns(ann_file)

	class_dist_by_id = {}
	for ann in anns['annotations']:
		cat_id = str(ann['category_id'])
		class_dist_by_id[cat_id] = class_dist_by_id.get(cat_id, 0) + 1

	id_name_map = {}
	for cat in anns['categories']:
		id_name_map[str(cat['id'])] = cat['name']

	class_dist_by_name = {}
	for cat_id in class_dist_by_id:
		count = class_dist_by_id[cat_id]
		cat_name = id_name_map[cat_id]
		class_dist_by_name[cat_name] = class_dist_by_name.get(cat_name, 0) + count

	return class_dist_by_name

def _update_counts(class_dist, partial_dist):
	for classname in partial_dist:
		class_dist[classname] = class_dist.get(classname, 0) + partial_dist[classname]

def get_anns_for_img(img_file, anns):
	img_desc = _get_img_desc(img_file, anns)
	if img_desc == None:
		return []
	
	relevant_anns = _filter_by_img_id(img_desc['id'], anns)

	return relevant_anns

def _get_img_desc(img_file, anns):
	requested_img_desc = None
	for img_desc in anns['images']:
		if img_desc['file_name'] == img_file.name:
			requested_img_desc = img_desc
			break
	return requested_img_desc

def _filter_by_img_id(img_id, anns):
	relevant_anns = []
	for ann in anns['annotations']:
		if ann['image_id'] == img_id:
			relevant_anns.append(ann)
	return relevant_anns

def get_img_dimensions(img_file, anns):
	img_desc = _get_img_desc(img_file, anns)
	if img_desc == None:
		return (0, 0)

	return (img_desc['height'], img_desc['width'])