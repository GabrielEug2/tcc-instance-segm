import argparse
import json
from pathlib import Path

def compute_class_dist(ann_file):
	"""Computes the class dist for the specified file.

	Args:
		ann_file (Path): path to an annotation file, in COCO format

	Raises:
		FileNotFoundError: if the requested file was not found
		ValueError: if the annotations does not follow the required format

	Returns:
		dict[str, int]: class distribution, by name
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


if __name__ == '__main__':
	parser = argparse.ArgumentParser("Computes the class distribution on COCO")
	parser.add_argument('ann_dir', help='Directory where you placed the annotations')

	args = parser.parse_args()

	ANN_FILENAMES = [
		'instances_train2017.json',
		'instances_val2017.json',
	]
	ann_files = [Path(args.ann_dir, fn) for fn in ANN_FILENAMES]

	class_dists = {}
	for ann_file in ann_files:
		print(f"Processing {ann_file.name}...")
		out_name = str(ann_file.name).replace('instances_', '').replace('2017.json', '')

		class_dists[out_name] = compute_class_dist(ann_file)

	print(f"Agreggating results...")
	sum_of_dists = {}
	for name in class_dists:
		file_dist = class_dists[name]
		for class_name in file_dist:
			sum_of_dists[class_name] = sum_of_dists.get(class_name, 0) + file_dist[class_name]
	class_dists['train-val'] = sum_of_dists

	for name in class_dists:
		dist = class_dists[name]
		dist = dict(sorted(dist.items(), key=lambda c: c[1], reverse=True))

		out_file = Path(__file__).parent / f"classdist_{name}.json"
		with out_file.open('w') as f:
			json.dump(dist, f, indent=4)