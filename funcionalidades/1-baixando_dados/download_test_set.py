import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description='Downloads data from the OpenImages dataset.')
parser.add_argument('n_imgs', type=int, help='Number of images to download')
parser.add_argument('out_dir', help='Directory to save')
args = parser.parse_args()

n_imgs = args.n_imgs

out_dir = Path(args.out_dir)
if not out_dir.exists():
	out_dir.mkdir(parents=True)
else:
	op = input((f'out_dir "{str(out_dir)}" exists. Do you want '
	             'to overwrite it? [y/n] ')).strip().lower()
	if op != 'y':
		print('Operation cancelled.')
		exit()


COCO_CLASS_DIST_FILE = Path(__file__).parent / 'coco_classdist.json'
with COCO_CLASS_DIST_FILE.open('r') as f:
	coco_class_dist = json.load(f)

dist_sorted_by_count = dict(sorted(
	coco_class_dist.items(),
	key=lambda c: c[1],
	reverse=True
))
wanted_classes_by_order = [c.lower() for c in dist_sorted_by_count]

# Late import just to make sure --help is fast
from segm_lib.misc import download_utils
download_utils.download_from_openimages(wanted_classes_by_order, n_imgs, out_dir)