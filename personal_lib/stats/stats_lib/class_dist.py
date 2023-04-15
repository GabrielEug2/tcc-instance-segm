from pathlib import Path
import json

VALID_FILE_TYPES = ['annotations', 'predictions']

def class_distribution(ann_or_pred_files: Path|list[Path], file_type: str):
	"""Computes the class distribution on the specified files.

	Args:
		ann_or_pred_files (Path): path to the files you want to compute
			the class distribution on.
		file_type (str): format of the content. Must be either: "annotations",
			if they are annotation files in COCO format, or "predictions",
			if they are prediction files outputed by the inference_lib

	Raises:
		FileNotFoundError: if any of the requested files was not found.
		ValueError: if an invalid file_type was given, or the content does
			not follow the expected format.

	Returns:
		dict[str, int]: class distribution, by name.
	"""
	if type(ann_or_pred_files) == Path:
		ann_or_pred_files = [ann_or_pred_files]
	try:
		_validate_arguments(ann_or_pred_files, file_type)
	except (ValueError, FileNotFoundError):
		raise

	class_dist = _compute_class_dist(ann_or_pred_files, file_type)

	return class_dist

def _validate_arguments(ann_or_pred_files, file_type):
	for f in ann_or_pred_files:
		if not f.exists():
			raise FileNotFoundError(str(f))

	if file_type not in VALID_FILE_TYPES:
		raise ValueError("Unknown file_type")
	
def _compute_class_dist(ann_or_pred_files: list[Path], file_type):
	class_dist = {}
	if file_type == 'annotations':
		parsing_func = _dist_from_ann
	else:
		parsing_func = _dist_from_pred

	for file in ann_or_pred_files:
		print(f"Processing {file.name}...") # Só pra saber se é normal a demora, tipo o annotations_train
		try:
			file_dist = parsing_func(file)
		except Exception as e:
			raise ValueError(f"File \"{str(file)}\" does not follow the expected format") from e
		_update_counts(class_dist, file_dist)

	return class_dist

def _dist_from_ann(ann_file: Path):
	with ann_file.open('r') as f:
		anns = json.load(f)

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

def _dist_from_pred(pred_file: Path):
	with pred_file.open('r') as f:
		preds = json.load(f)

	class_dist = {}
	for pred in preds:
		classname = pred['classname']
		class_dist[classname] = class_dist.get(classname, 0) + 1

	return class_dist

def _update_counts(class_dist, file_dist):
	for classname in file_dist:
		n_ocurrences_on_file = file_dist[classname]
		class_dist[classname] = class_dist.get(classname, 0) + n_ocurrences_on_file