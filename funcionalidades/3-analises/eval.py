import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Evaluates the results')
parser.add_argument('pred_dir', help='Directory containing the predictions')
parser.add_argument('ann_dir', help='Directory containing the annotations')
parser.add_argument('possible_classes_dir', help='Directory containing a list of possible classes for each model')
parser.add_argument('coco_ann_file', help='File containing the *original* annotations, in COCO-format')
parser.add_argument('img_dir', help=('Directory containing the images the predictions and '
                    'annotations refer to, to plot true positives, false positives and '
                    'false negatives'))
parser.add_argument('out_dir', help='directory to save the results')
parser.add_argument('-y', '--overwrite', action='store_true')
args = parser.parse_args()

pred_dir = Path(args.pred_dir)
if not pred_dir.exists():
	raise FileNotFoundError(str(pred_dir))

ann_dir = Path(args.ann_dir)
if not ann_dir.exists():
	raise FileNotFoundError(str(ann_dir))

possible_classes_dir = Path(args.possible_classes_dir)
if not possible_classes_dir.exists():
	raise FileNotFoundError(str(possible_classes_dir))

coco_ann_file = Path(args.coco_ann_file)
if not coco_ann_file.exists():
	raise FileNotFoundError(str(coco_ann_file))

img_dir = Path(args.img_dir)
if not img_dir.exists():
	raise FileNotFoundError(str(img_dir))

out_dir = Path(args.out_dir)
if out_dir.exists() and not args.overwrite:
	op = input((f'out_dir "{str(out_dir)}" exists. Do you want '
	             'to overwrite it? [y/n] ')).lower()
	if op != 'y':
		print('Operation cancelled.')
		exit()


# Import depois pro --help ser rápido
from segm_lib.eval import evaluate_all
evaluate_all(pred_dir, ann_dir, possible_classes_dir, coco_ann_file, img_dir, out_dir)