
import fiftyone as fo
import fiftyone.zoo as foz

import argparse

def download(out_dir):
	dataset = foz.load_zoo_dataset(
		"open-images-v6",
		split="validation",
		label_types="segmentations",
		classes=["Person", "Car", "Book"],
		max_samples=200
	)

	label_field = "ground_truth"
	dataset_type = fo.types.COCODetectionDataset

	dataset.export(
		export_dir=out_dir,
		dataset_type=dataset_type,
		label_field=label_field
	)

parser = argparse.ArgumentParser()
parser.add_argument('out_dir')

args = parser.parse_args()

download(args.out_dir)