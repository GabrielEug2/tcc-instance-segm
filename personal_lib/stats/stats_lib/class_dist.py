import json
from pathlib import Path

def compute_class_dist(ann_file: Path):
	"""Computes the class dist for the specified file.

	Args:
		ann_file (Path): path to an annotation file, in COCO format.

	Raises:
		FileNotFoundError: if the requested file was not found.
		ValueError: if the annotations does not follow the required format.

	Returns:
		dict[str, int]: class distribution, by name.
	"""
	if not ann_file.exists():
		raise FileNotFoundError(str(f))
	
	class_dist_by_id = {}
	id_name_map = {}
	try:
		with ann_file.open('r') as f:
			anns = json.load(f)

		for ann in anns['annotations']:
			cat_id = ann['category_id']
			class_dist_by_id[str(cat_id)] = class_dist_by_id.get(str(cat_id), 0) + 1

		for cat in anns['categories']:
			cat_id = cat['id']
			if str(cat_id) not in id_name_map:
				id_name_map[str(cat_id)] = cat['name']
	except Exception as e:
		raise ValueError(f"File \"{str(ann_file)}\" does not follow the required format") from e

	# Eu quero por nome, não por id
	class_dist_by_name = {}
	for cat_id in id_name_map:
		class_dist_by_name[id_name_map[cat_id]] = class_dist_by_id[cat_id]

	return class_dist_by_name