import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Evaluates the results')
parser.add_argument('pred_dir', help='directory containing the predictions')
parser.add_argument('ann_dir', help='directory containing the *split* annotations')
parser.add_argument('ann_file', help='file containing the *original* annotations')
parser.add_argument('out_dir', help='directory to save the results')
parser.add_argument('-y', '--overwrite', action='store_true')
args = parser.parse_args()

pred_dir = Path(args.pred_dir)
ann_dir = Path(args.ann_dir)
ann_file = Path(args.ann_file)
out_dir = Path(args.out_dir)

if not pred_dir.exists():
	raise FileNotFoundError(str(pred_dir))
if not ann_dir.exists():
	raise FileNotFoundError(str(ann_dir))
if not ann_file.exists():
	raise FileNotFoundError(str(ann_file))

if not out_dir.exists():
	out_dir.mkdir(parents=True)
elif args.overwrite:
	pass
else:
	op = input(f"out_dir \"{str(out_dir)}\" exists. Do you want to overwrite it? [y/n] ").strip().lower()
	if op != 'y':
		print("Exiting")
		exit()

# Import depois pro --help ser rápido
from segm_lib.eval import evaluate_all
evaluate_all(pred_dir, ann_dir, ann_file, out_dir)