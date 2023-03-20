
import argparse

import fiftyone as fo
import fiftyone.zoo as foz

def download(out_dir):
    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        split="validation",
        image_ids=[
            '0aca4dabcb743f3b',
            '0b1e7c2be07dcfed',
            '0b30ad1afeba9bca',
            '0b9aec984321bf53'
        ],
        label_types="segmentations"
    )

    print(dataset)

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