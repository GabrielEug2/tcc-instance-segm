import json
from pathlib import Path

def compute_class_dist(pred_files: list[Path]):
	class_dist = {}

	for pred_file in pred_files:
		try:
			class_dist_on_file = extract_info(pred_file)
		except Exception as e:
			raise ValueError(f"File \"{str(pred_file)}\" does not follow the expected format") from e
		update_counts(class_dist, class_dist_on_file)

	return class_dist

def extract_info(pred_file: Path):
	print(f"Processing {pred_file.name}...") # Só pra saber se é normal a demora, tipo o annotations_train
	with pred_file.open('r') as f:
		preds = json.load(f)

	class_dist = {}
	for pred in preds:
		classname = pred['classname']
		class_dist[classname] = class_dist.get(classname, 0) + 1

	return class_dist

def update_counts(class_dist, parcial_dist):
	for classname in parcial_dist:
		count = parcial_dist[classname]
		class_dist[classname] = class_dist.get(classname, 0) + count