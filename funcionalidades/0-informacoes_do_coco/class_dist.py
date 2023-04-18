import argparse
import json
from pathlib import Path

from personal_lib import ann_parser

parser = argparse.ArgumentParser()
parser.add_argument('ann_dir', help='Directory where you placed the annotations')
args = parser.parse_args()

ANN_FILENAMES = ['instances_train2017.json', 'instances_val2017.json']
ann_files = [Path(args.ann_dir, fn) for fn in ANN_FILENAMES]
class_dist = ann_parser.class_distribution(ann_files)

out_file = Path(__file__).parent / f"coco_classdist.json"
sorted_dist = dict(sorted(class_dist.items(), key=lambda c: c[1], reverse=True))
with out_file.open('w') as f:
	json.dump(sorted_dist, f, indent=4)