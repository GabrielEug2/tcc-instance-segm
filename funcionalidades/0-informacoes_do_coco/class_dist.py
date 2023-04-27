import argparse
import json
from pathlib import Path

from personal_lib.parsing.annotations import class_dist_from_multiple_files

parser = argparse.ArgumentParser()
parser.add_argument('ann_dir', help='Directory where you placed the annotations')
args = parser.parse_args()

ANN_FILENAMES = ['instances_train2017.json', 'instances_val2017.json']
ann_files = [Path(args.ann_dir, fn) for fn in ANN_FILENAMES]

class_dist = class_dist_from_multiple_files(ann_files)

out_file = Path(__file__).parent / f"coco_classdist.json"
dist_sorted_by_count = dict(sorted(class_dist.items(), key=lambda c: c[1], reverse=True))
with out_file.open('w') as f:
	json.dump(dist_sorted_by_count, f, indent=4)