import argparse
from pathlib import Path

from personal_lib.eval import evaluate_all

parser = argparse.ArgumentParser(description='Evaluates the results')
parser.add_argument('ann_file', help='file containing the annotations')
parser.add_argument('pred_dir', help='directory containing the predictions')
args = parser.parse_args()

ann_file = Path(args.ann_file)
pred_dir = Path(args.pred_dir)

evaluate_all(ann_file, pred_dir)