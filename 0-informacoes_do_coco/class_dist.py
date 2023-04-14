import argparse
import json
from pathlib import Path

import stats_lib

parser = argparse.ArgumentParser()
parser.add_argument('ann_dir', help='Directory where you placed the annotations')

args = parser.parse_args()

ANN_FILENAMES = ['instances_train2017.json', 'instances_val2017.json']
ann_files = [Path(args.ann_dir, fn) for fn in ANN_FILENAMES]
class_dist = stats_lib.class_distribution(ann_files, type='annotations')

class_dist = dict(sorted(class_dist.items(), key=lambda c: c[1], reverse=True))
out_file = Path(__file__).parent / f"coco_classdist.json"
with out_file.open('w') as f:
	json.dump(class_dist, f, indent=4)