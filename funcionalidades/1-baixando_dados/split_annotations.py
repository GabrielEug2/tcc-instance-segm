import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('ann_file', help='file containing the COCO annotations')
parser.add_argument('out_dir', help='directory to save the custom annotations')
args = parser.parse_args()

ann_file = Path(args.ann_file)
out_dir = Path(args.out_dir)

if not ann_file.exists():
	raise FileNotFoundError(str(ann_file))
if not out_dir.exists():
	out_dir.mkdir(parents=True)
else:
	op = input(f"out_dir \"{str(out_dir)}\" exists. Do you want to overwrite it? [y/n] ").strip().lower()
	if op != 'y':
		print("Exiting")
		exit()

# Import depois pro --help ser rápido
from segm_lib.ann_manager import AnnManager
AnnManager(out_dir).from_coco_file(ann_file)