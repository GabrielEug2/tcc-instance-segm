import argparse
from pathlib import Path

import fiftyone as fo

from view_lib import load_data

SAMPLE_DATA_DIR = Path(__file__).parent / 'sample_data'

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_dir', help='directory containing the images and annotations',
                    default=str(SAMPLE_DATA_DIR / 'dataset'))
parser.add_argument('-p', '--predictions_dir', help='directory containing the predictions',
                    default=str(SAMPLE_DATA_DIR / 'predictions'))

args = parser.parse_args()

dataset = load_data(args.dataset_dir, args.predictions_dir)

session = fo.launch_app(dataset)
session.wait()

dataset.draw_labels('tmp', label_fields=['ground_truth_segmentations', 'maskrcnn_pred'])
