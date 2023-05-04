import argparse
from pathlib import Path

from segm_lib.eval import evaluate_all

parser = argparse.ArgumentParser(description='Evaluates the results')
parser.add_argument('pred_dir', help='directory containing the predictions')
parser.add_argument('ann_dir', help='directory containing the annotations')
parser.add_argument('out_dir', help='directory to save the results')
args = parser.parse_args()

pred_dir = Path(args.pred_dir)
ann_dir = Path(args.ann_dir)
out_dir = Path(args.out_dir)

evaluate_all(pred_dir, ann_dir, out_dir)