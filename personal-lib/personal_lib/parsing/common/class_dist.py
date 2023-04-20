
from pathlib import Path

VALID_FILETYPES = ['annotations', 'predictions']

def class_distribution(ann_or_pred_files: Path|list[Path], filetype: str):
	if filetype not in VALID_FILETYPES:
		raise ValueError(f"Invalid filetype: \"{filetype}\". Must be one of {VALID_FILETYPES}")
	for f in ann_or_pred_files:
		if not f.exists():
			raise FileNotFoundError(str(f))

	if type(ann_or_pred_files) == Path:
		ann_or_pred_files = [ann_or_pred_files]

	class_dist = _class_dist_from_files(ann_or_pred_files, filetype)

	return class_dist

def _class_dist_from_files(ann_or_pred_files, filetype):
	if filetype == 'annotations':
		file_dist_func = _class_dist_from_ann_file
	if filetype == 'predictions': 
		file_dist_func = _class_dist_from_pred_file

	total_class_dist = {}
	for ann_or_pred_file in ann_or_pred_files:
		print(f"Processing {ann_or_pred_file.name}...") # Só pra saber se é normal a demora, tipo o annotations_train
		try:
			file_dist = file_dist_func(ann_or_pred_file)
		except Exception as e:
			raise ValueError(f"File \"{str(ann_or_pred_file)}\" does not follow the expected format") from e

		_update_counts(total_class_dist, file_dist)

	return total_class_dist

def _class_dist_from_ann_file(ann_file: Path) -> dict:
	from personal_lib.parsing.annotations import Annotations

	anns = Annotations(ann_file)
	return anns.class_distribution()
	
def _class_dist_from_pred_file(pred_file: Path) -> dict:
	from personal_lib.parsing.predictions import Predictions

	preds = Predictions.load_from_file(pred_file)
	return preds.class_distribution()

def _update_counts(total_class_dist, partial_dist):
	for classname in partial_dist:
		total_class_dist[classname] = total_class_dist.get(classname, 0) + partial_dist[classname]