import argparse
import json
from pathlib import Path

import stats_lib

parser = argparse.ArgumentParser()
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
	class_dist = stats_lib.compute_class_dist(ann_file)

	out_name = str(ann_file.name).removeprefix('instances_').removesuffix('2017.json')
	class_dists[out_name] = class_dist

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