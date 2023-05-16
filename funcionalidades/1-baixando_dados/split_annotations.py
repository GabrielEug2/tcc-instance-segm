import argparse
from pathlib import Path

from segm_lib.ann_manager import AnnManager

parser = argparse.ArgumentParser()
parser.add_argument('ann_file', help='file containing the COCO annotations')
parser.add_argument('out_dir', help='directory to save the custom annotations')
args = parser.parse_args()

ann_file = Path(args.ann_file)
out_dir = Path(args.out_dir)

out_dir.mkdir(parents=True, exist_ok=True)
AnnManager(out_dir).from_coco_file(ann_file)