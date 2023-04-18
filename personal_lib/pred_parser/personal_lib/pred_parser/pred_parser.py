import json
from pathlib import Path

import personal_lib.mask_conversions as mask_conversions

# Prediction format:
# 	same as personal_lib.inference.predictors.base_predictor,
#	but with RLE instead

def load_preds(pred_file: Path):
	if not pred_file.exists():
		raise FileNotFoundError(str(pred_file))

	try:
		with pred_file.open('r') as f:
			preds = json.load(f)

		for pred in preds:
			# É salvo como RLE, mas para usar é melhor binário
			pred['mask'] = mask_conversions.rle_to_bin_mask(pred['mask'])
	except json.JSONDecodeError as e:
		raise ValueError(f"File \"{str(pred_file)}\" does not follow the required format") from e

	return preds

def class_distribution(pred_files: Path|list[Path]):
	"""Computes the class distribution on the specified files.

	Args:
		pred_files (Path): path to the files you want to compute the class
			distribution on.

	Raises:
		FileNotFoundError: if any of the requested files was not found.
		ValueError: if the content does	not follow the expected format.

	Returns:
		dict[str, int]: class distribution, by name.
	"""
	for f in pred_files:
		if not f.exists():
			raise FileNotFoundError(str(f))
	
	if type(pred_files) == Path:
		pred_files = [pred_files]

	class_dist = _compute_class_dist(pred_files)

	return class_dist

def _compute_class_dist(pred_files: list[Path]):
	class_dist = {}

	for pred_file in pred_files:
		print(f"Processing {pred_file.name}...") # Só pra saber se é normal a demora, tipo o annotations_train
		try:
			preds = load_preds(pred_file)
			file_dist = _class_dist(preds)
		except Exception as e:
			raise ValueError(f"File \"{str(pred_file)}\" does not follow the expected format") from e

		_update_counts(class_dist, file_dist)

	return class_dist

def _class_dist(preds):
	class_dist = {}

	for pred in preds:
		classname = pred['classname']
		class_dist[classname] = class_dist.get(classname, 0) + 1

	return class_dist

def _update_counts(class_dist, partial_dist):
	for classname in partial_dist:
		class_dist[classname] = class_dist.get(classname, 0) + partial_dist[classname]