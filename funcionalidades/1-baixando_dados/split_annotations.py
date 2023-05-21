import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('coco_ann_file', help='file containing the COCO annotations')
parser.add_argument('custom_ann_dir', help='directory to save the custom annotations')
args = parser.parse_args()

coco_ann_file = Path(args.coco_ann_file)
if not coco_ann_file.exists():
	raise FileNotFoundError(str(coco_ann_file))

custom_ann_dir = Path(args.custom_ann_dir)
if custom_ann_dir.exists():
	op = input((f'custom_ann_dir "{str(custom_ann_dir)}" exists. Do you want '
	             'to overwrite it? [y/n] ')).strip().lower()
	if op != 'y':
		print('Operation cancelled.')
		exit()

# Import depois pro --help ser rápido
from segm_lib.core.managers.ann_manager import AnnManager
AnnManager(custom_ann_dir).from_coco_file(coco_ann_file)