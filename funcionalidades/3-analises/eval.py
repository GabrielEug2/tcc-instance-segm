import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Evaluates the results')
parser.add_argument('pred_dir', help='Directory containing the predictions')
parser.add_argument('ann_dir', help='Directory containing the annotations')
parser.add_argument('coco_ann_file', help='File containing the *original* annotations, in COCO-format')
parser.add_argument('out_dir', help='directory to save the results')
parser.add_argument('-y', '--overwrite', action='store_true')
args = parser.parse_args()

pred_dir = Path(args.pred_dir)
if not pred_dir.exists():
	raise FileNotFoundError(str(pred_dir))

ann_dir = Path(args.ann_dir)
if not ann_dir.exists():
	raise FileNotFoundError(str(ann_dir))

coco_ann_file = Path(args.coco_ann_file)
if not coco_ann_file.exists():
	raise FileNotFoundError(str(coco_ann_file))

out_dir = Path(args.out_dir)
if out_dir.exists() and not args.overwrite:
	op = input((f'out_dir "{str(out_dir)}" exists. Do you want '
	             'to overwrite it? [y/n] ')).lower()
	if op != 'y':
		print('Operation cancelled.')
		exit()

# Import depois pro --help ser rápido
from segm_lib.eval import evaluate_all
evaluate_all(pred_dir, ann_dir, coco_ann_file, out_dir)