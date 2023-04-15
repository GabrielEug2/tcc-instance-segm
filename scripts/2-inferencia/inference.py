import argparse
from pathlib import Path

from inference_lib import run_inference

parser = argparse.ArgumentParser(
	description='Runs instance segmentation on a set of images and save '
				'the results on the specified folder'
)
parser.add_argument('img_file_or_dir', help='path to an image or dir of images to segment.')
parser.add_argument('out_dir', help='directory to save the results')
parser.add_argument('-c', '--compressed-masks', action='store_true',
					help='whether or not to compress masks for smaller file sizes')
parser.add_argument('-m', '--save-masks', action='store_true',
					help='whether or not to save individual masks as images')

args = parser.parse_args()

img_file_or_dir = Path(args.img_file_or_dir)
out_dir = Path(args.out_dir)
run_inference(img_file_or_dir, out_dir,
    compressed_masks=args.compressed_masks,
    save_masks=args.save_masks,
)