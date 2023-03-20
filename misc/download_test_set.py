
import argparse

import fiftyone as fo
import fiftyone.zoo as foz

def download(n_imgs, out_dir):
	dataset = foz.load_zoo_dataset(
		"open-images-v6",
		split="validation",
		label_types="segmentations",
		classes=["Person", "Car", "Book"],
		max_samples=n_imgs
	)

	dataset.export(
		export_dir=out_dir,
		dataset_type=fo.types.COCODetectionDataset
	)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('n_imgs', help='Number of images to download')
	parser.add_argument('out_dir', help='Directory to save')

	args = parser.parse_args()

	download(args.n_imgs, args.out_dir)
