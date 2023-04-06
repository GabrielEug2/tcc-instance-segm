
import argparse
import json
from pathlib import Path

import fiftyone as fo
import fiftyone.utils.openimages as openimages
import fiftyone.zoo as fozoo

COCO_CLASS_DIST_FILE = Path(__file__).parent / 'classdist_coco.json'

def filter_common_classes():
	"""Filter classes that exist on both datasets (COCO and Openimages).

	Returns:
		list: list of class names, sorted by n_ocurrences on COCO
	"""
	with COCO_CLASS_DIST_FILE.open('r') as f:
		coco_class_dist = json.load(f)

	coco_classes = coco_class_dist.keys()
	openimages_classes = [x.lower() for x in openimages.get_segmentation_classes()]
	common_classes = [x for x in coco_classes if x in openimages_classes]

	sorted_dist = sorted(coco_class_dist.items(), key=lambda c: c[1], reverse=True)
	sorted_classes = [c[0] for c in sorted_dist if c[0] in common_classes]
	
	return sorted_classes

def download(n_imgs, out_dir, classes, only_matching=False):
	"""Download a set of images from Openimages, and their respective annotations.

	Args:
		n_imgs (int): number of images to download
		out_dir (Path): directory to save the data
		classes (list[str]): _description_. list of classes, to
			only download images with these specific classes.
		only_matching (bool): whether to only download labels that match the 
			classes you requested (True), or to load all labels for samples
			that have the classes you requested (False). Default to False.
	"""
	classes = [x.capitalize() for x in classes]

	dataset = fozoo.load_zoo_dataset(
		"open-images-v6",
		split="validation",
		label_types="segmentations",
		shuffle=True,
		max_samples=n_imgs,
		classes=classes,
		only_matching=only_matching,
	)

	out_dir = Path(out_dir)
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

	# Quero só imagens com as 10 classes que mais tem no COCO
	common_classes = filter_common_classes()
	top_classes = common_classes[:10]

	download(args.n_imgs, args.out_dir, classes=top_classes)