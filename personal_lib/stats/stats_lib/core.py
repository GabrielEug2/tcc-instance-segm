from pathlib import Path

from . import ann_logic
from . import pred_logic

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

	for f in ann_or_pred_files:
		if not f.exists():
			raise FileNotFoundError(str(f))

	if file_type == 'annotations':
		module = ann_logic
	elif file_type == 'predictions':
		module = pred_logic
	else:
		raise ValueError("Unknown file_type")

	try:
		class_dist = module.compute_class_dist(ann_or_pred_files)
	except ValueError:
		raise

	return class_dist