import json
from pathlib import Path

def compute_classmap(ann_file: Path):
	"""Computes the classmap for the specified file.

	Args:
		ann_file (Path): path to an annotation file, in COCO format.

	Raises:
		FileNotFoundError: if the requested file was not found.
		ValueError: if the annotations does not follow the required format.

	Returns:
		dict[str, dict[str, str]]: maps associating class ids with class names.
			"default" is the raw IDs from the file, and "model" is the
			more commonly used normalization [0,N).
	"""
	if not ann_file.exists():
		raise FileNotFoundError(str(f))
	
	default_coco_map = {}
	try:
		with open(ann_file) as f:
			anns = json.load(f)

		for cat in anns["categories"]:
			default_coco_map[cat['id']] = cat['name']
	except Exception as e:
		raise ValueError(f"File \"{str(ann_file)}\" does not follow the required format") from e

	model_map = {}
	n_classes = len(default_coco_map.keys())
	sorted_cats = sorted(default_coco_map.items(), key=lambda c: int(c[0]))
	for i in range(n_classes):
		model_map[str(i)] = sorted_cats[i][1]

	return { 'default': default_coco_map, 'model': model_map }