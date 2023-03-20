
# https://docs.voxel51.com/user_guide/evaluation.html

import argparse

import fiftyone as fo

def build_parser():
    parser = argparse.ArgumentParser(
        description=''
    )
    parser.add_argument('dataset_dir', help='directory containing the images and annotations')
    #default sample
    parser.add_argument('predictions_dir', help='directory containing the predictions')
    #default sample
    return parser

if __name__ == '__main__':
    print('oi')
    parser = build_parser()
    args = parser.parse_args()

    name = "my-dataset"
    dataset_type = fo.types.COCODetectionDataset

    dataset = fo.Dataset.from_dir(
        dataset_dir=args.dataset_dir,
        dataset_type=dataset_type,
        name=name,
    )

    session = fo.launch_app(dataset)