import argparse
from collections import defaultdict
from pathlib import Path
import json
from segm_lib.coco_annotations import COCOAnnotations

parser = argparse.ArgumentParser()
parser.add_argument('ann_dir', help='Directory where you placed the annotations')
args = parser.parse_args()

ANN_FILENAMES = ['instances_train2017.json', 'instances_val2017.json']
ann_files = [Path(args.ann_dir, fn) for fn in ANN_FILENAMES]
for f in ann_files:
	if not f.exists():
		raise FileNotFoundError(str(f))

total_class_dist = defaultdict(lambda: 0)
for ann_file in ann_files:
	print(f"Processing {ann_file.name}...") # Só pra saber se é normal a demora, tipo o instances_train.2017
	try:
		file_dist = COCOAnnotations(ann_file).class_distribution()
	except FileNotFoundError as e:
		raise
	except Exception as e:
		raise ValueError(f"File \"{str(ann_file)}\" doesn't follow the expected format") from e

	for classname in file_dist:
		total_class_dist[classname] += file_dist[classname]

dist_sorted_by_count = dict(sorted(total_class_dist.items(), key=lambda c: c[1], reverse=True))
out_file = Path(__file__).parent / f"coco_classdist.json"
out_file.parent.mkdir(parents=True, exist_ok=True)
with out_file.open('w') as f:
	json.dump(dist_sorted_by_count, f, indent=4)