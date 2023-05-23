import random
from pathlib import Path


def download_from_openimages(class_list: list[str], n_imgs: int, out_dir: Path):
	"""Downloads data from the OpenImages dataset.

	Args:
		class_list (list[str]): desired classes, by importance order.
		n_imgs (int): number of images to download.
		out_dir (Path): dir to save the data.
	"""
	import fiftyone.utils.openimages as openimages
	import fiftyone.zoo as fozoo
	from fiftyone.types import COCODetectionDataset

	normalized_class_list = [c.capitalize() for c in class_list]
	openimages_classes = openimages.get_segmentation_classes() # openimage classes are capitalized
	common_classes = [c for c in normalized_class_list if c in openimages_classes]

	classes_to_download = common_classes[:10]

	dataset = fozoo.load_zoo_dataset(
		'open-images-v6',
		split='validation',
		label_types='segmentations',
		max_samples=n_imgs,
		classes=classes_to_download,
		shuffle=True,
		seed=random.randrange(0, 1000),
	)
	out_dir.mkdir(parents=True, exist_ok=True)
	dataset.export(
		export_dir=str(out_dir),
		dataset_type=COCODetectionDataset
	)
	Path.rename(out_dir / 'data', out_dir / 'images')
	Path.rename(out_dir / 'labels.json', out_dir / 'annotations.json')