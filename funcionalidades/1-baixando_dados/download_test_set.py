
import argparse
import json
from pathlib import Path
import random

# Os imports do fiftyone não estão aqui porque é muito lento.
# Se deixar no topo da arquivo, o --help também demora

COCO_CLASS_DIST_FILE = Path(__file__).parent / 'coco_classdist.json'

def filter_common_classes():
	"""Filter classes that exist on both datasets (COCO and Openimages).

	Returns:
		list: list of class names, sorted by n_ocurrences on COCO.
	"""
	with COCO_CLASS_DIST_FILE.open('r') as f:
		coco_class_dist = json.load(f)

	import fiftyone.utils.openimages as openimages

	coco_classes = coco_class_dist.keys()
	openimages_classes = [x.lower() for x in openimages.get_segmentation_classes()]
	common_classes = [x for x in coco_classes if x in openimages_classes]

	filtered_dist = { name: count for name, count in coco_class_dist.items() if name in common_classes }
	dist_sorted_by_count = sorted(filtered_dist.items(), key=lambda c: c[1], reverse=True)
	classnames_sorted_by_count = [c[0] for c in dist_sorted_by_count]
	
	return classnames_sorted_by_count

def download(n_imgs: int, out_dir: Path, classes: list[str], only_matching=False):
	"""Download a set of images from Openimages, and their respective annotations.

	Args:
		n_imgs (int): number of images to download,
		out_dir (Path): directory to save the data,
		classes (list[str]): list of classes, to only download images that
			contains objects of said classes.
		only_matching (bool): whether to only download labels that match the 
			classes you requested (True) or download all labels for images
			that have the classes you requested (False). Default to False.
	"""
	import fiftyone as fo
	import fiftyone.zoo as fozoo

	dataset = fozoo.load_zoo_dataset(
		"open-images-v6",
		split="validation",
		label_types="segmentations",
		max_samples=n_imgs,
		classes=classes,
		shuffle=True,
		seed=random.randrange(0, 1000),
		only_matching=only_matching,
	)

	dataset.export(
		export_dir=str(out_dir),
		dataset_type=fo.types.COCODetectionDataset
	)
	Path.rename(out_dir / 'data', out_dir / 'images')
	Path.rename(out_dir / 'labels.json', out_dir / 'annotations.json')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('n_imgs', type=int, help='Number of images to download')
	parser.add_argument('out_dir', help='Directory to save')
	args = parser.parse_args()

	common_classes = filter_common_classes()
	top_classes = common_classes[:10]

	out_dir = Path(args.out_dir)
	top_classes = [x.capitalize() for x in top_classes]
	download(args.n_imgs, out_dir, classes=top_classes)