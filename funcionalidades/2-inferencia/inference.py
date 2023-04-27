import argparse
from pathlib import Path

from personal_lib.inference import Inference

parser = argparse.ArgumentParser(
	description='Runs instance segmentation on a set of images and save '
				'the results on the specified folder'
)
parser.add_argument('img_file_or_dir', help='path to an image or dir of images to segment.')
parser.add_argument('out_dir', help='directory to save the results')
args = parser.parse_args()

img_file_or_dir = Path(args.img_file_or_dir)
out_dir = Path(args.out_dir)

# valid_models = Inference.VALID_MODELS
inference = Inference
# Inference.set_models()

run_inference(img_file_or_dir, out_dir)