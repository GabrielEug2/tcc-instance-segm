import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Runs instance segmentation on a set of images.')
parser.add_argument('img_file_or_dir', help='Image or directory of images to segment.')
parser.add_argument('out_dir', help='Directory to save the results.')
args = parser.parse_args()

img_file_or_dir = Path(args.img_file_or_dir)
if not img_file_or_dir.exists():
	raise FileNotFoundError(str(img_file_or_dir))

out_dir = Path(args.out_dir)
if not out_dir.exists():
	out_dir.mkdir(parents=True)
else:
	op = input((f'out_dir "{str(out_dir)}" exists. Do you want '
	             'to overwrite it? [y/n] ')).strip().lower()
	if op != 'y':
		print('Operation cancelled.')
		exit()


# Import depois pro --help ser rápido
from segm_lib.inference import run_inference
run_inference(img_file_or_dir, out_dir)