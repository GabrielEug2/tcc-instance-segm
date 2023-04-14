import json
from pathlib import Path

def compute_class_dist(ann_files: list[Path]):
	class_dist = {}

	for ann_file in ann_files:
		try:
			ann_data = extract_info(ann_file)
		except Exception as e:
			raise ValueError(f"File \"{str(ann_file)}\" does not follow the expected format") from e
		update_counts(class_dist, ann_data)

	return class_dist

def extract_info(ann_file: Path):
	print(f"Processing {ann_file.name}...") # Só pra saber se é normal a demora, tipo o annotations_train
	with ann_file.open('r') as f:
		anns = json.load(f)

	class_dist_by_id = {}
	for ann in anns['annotations']:
		cat_id = str(ann['category_id'])
		class_dist_by_id[cat_id] = class_dist_by_id.get(cat_id, 0) + 1

	id_name_map = {}
	for cat in anns['categories']:
		id_name_map[str(cat['id'])] = cat['name']

	ann_data = {
		'class_dist_by_id': class_dist_by_id,
		'id_name_map': id_name_map
	}
	return ann_data

def update_counts(class_dist, ann_data):
	for cat_id in ann_data['class_dist_by_id']:
		count = ann_data['class_dist_by_id'][cat_id]
		cat_name = ann_data['id_name_map'][cat_id]
		class_dist[cat_name] = class_dist.get(cat_id, 0) + count