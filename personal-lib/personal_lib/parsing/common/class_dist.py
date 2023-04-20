
from pathlib import Path

from ..annotations import Annotations
from ..predictions import Predictions

FILETYPE_CLASS_MAP = {
	'annotations': Annotations,
	'predictions': Predictions,
}

def class_distribution(ann_or_pred_files: Path|list[Path], filetype: str):
	if filetype not in FILETYPE_CLASS_MAP:
		raise ValueError(f"Invalid filetype: \"{filetype}\". Must be one of {FILETYPE_CLASS_MAP.keys()}")

	for f in ann_or_pred_files:
		if not f.exists():
			raise FileNotFoundError(str(f))
	if type(ann_or_pred_files) == Path:
		ann_or_pred_files = [ann_or_pred_files]
	
	class_dist = _class_dist_from_files(ann_or_pred_files, filetype)

	return class_dist

def _class_dist_from_files(ann_or_pred_files: list[Path], filetype) -> dict:
	relevant_class = FILETYPE_CLASS_MAP[filetype]

	total_class_dist = {}
	for ann_or_pred_file in ann_or_pred_files:
		print(f"Processing {ann_or_pred_file.name}...") # Só pra saber se é normal a demora, tipo o annotations_train
		try:
			anns_or_preds = relevant_class(ann_or_pred_file)
			file_dist = anns_or_preds.class_distribution()
		except Exception as e:
			raise ValueError(f"File \"{str(ann_or_pred_file)}\" does not follow the expected format") from e

		_update_counts(total_class_dist, file_dist)

	return total_class_dist

def _update_counts(class_dist, partial_dist):
	for classname in partial_dist:
		class_dist[classname] = class_dist.get(classname, 0) + partial_dist[classname]