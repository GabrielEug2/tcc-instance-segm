"""Functions to parse predictions in personal_lib format.

See personal_lib.inference.predictors.base_predictor for details"""

import json
from pathlib import Path

from personal_lib.core import mask_conversions

def save_preds(predictions, out_file):
	for pred in predictions:
		pred['mask'] = mask_conversions.bin_mask_to_rle(pred['mask'])

	with out_file.open('w') as f:
		json.dump(predictions, f)

def load_preds(pred_file: Path):
	if not pred_file.exists():
		raise FileNotFoundError(str(pred_file))

	try:
		with pred_file.open('r') as f:
			predictions = json.load(f)

		for pred in predictions:
			pred['mask'] = mask_conversions.rle_to_bin_mask(pred['mask'])
	except json.JSONDecodeError as e:
		raise ValueError(f"File \"{str(pred_file)}\" does not follow the required format") from e

	return predictions

def class_distribution(pred_files: Path|list[Path]):
	for f in pred_files:
		if not f.exists():
			raise FileNotFoundError(str(f))
	
	if type(pred_files) == Path:
		pred_files = [pred_files]

	class_dist = _class_dist_from_files(pred_files)

	return class_dist

def _class_dist_from_files(pred_files: list[Path]) -> dict:
	total_class_dist = {}

	for pred_file in pred_files:
		print(f"Processing {pred_file.name}...") # Só pra saber se é normal a demora, tipo o annotations_train
		try:
			file_dist = _class_dist_from_file(pred_file)
		except Exception as e:
			raise ValueError(f"File \"{str(pred_file)}\" does not follow the expected format") from e

		_update_counts(total_class_dist, file_dist)

	return total_class_dist

def _class_dist_from_file(pred_file: Path) -> dict:
	preds = load_preds(pred_file)

	class_dist = {}

	for pred in preds:
		classname = pred['classname']
		class_dist[classname] = class_dist.get(classname, 0) + 1

	return class_dist

def _update_counts(class_dist, partial_dist):
	for classname in partial_dist:
		class_dist[classname] = class_dist.get(classname, 0) + partial_dist[classname]