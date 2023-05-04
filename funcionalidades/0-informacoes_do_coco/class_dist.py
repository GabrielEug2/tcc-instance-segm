import argparse
import json
from pathlib import Path

from personal_lib.coco_ann_parser import COCO_AnnParser

parser = argparse.ArgumentParser()
parser.add_argument('ann_dir', help='Directory where you placed the annotations')
args = parser.parse_args()

ANN_FILENAMES = ['instances_train2017.json', 'instances_val2017.json']
ann_files = [Path(args.ann_dir, fn) for fn in ANN_FILENAMES]

total_class_dist = {}
for ann_file in ann_files:
	print(f"Processing {ann_file.name}...") # Só pra saber se é normal a demora, tipo o annotations_train
	try:
		file_dist = COCO_AnnParser(ann_file).class_distribution()
	except Exception as e:
		raise ValueError(f"File \"{str(ann_file)}\" doesn't follow the expected format") from e

	for classname in file_dist:
		total_class_dist[classname] = total_class_dist.get(classname, 0) + file_dist[classname]

dist_sorted_by_count = dict(sorted(total_class_dist.items(), key=lambda c: c[1], reverse=True))
out_file = Path(__file__).parent / f"coco_classdist.json"
out_file.parent.mkdir(parents=True, exist_ok=True)
with out_file.open('w') as f:
	json.dump(dist_sorted_by_count, f, indent=4)