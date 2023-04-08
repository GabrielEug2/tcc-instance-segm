import argparse

import inference_lib

parser = argparse.ArgumentParser(
	description='Runs instance segmentation on a set of images and save '
				'the results on the specified folder'
)
parser.add_argument('img_file_or_dir', help='path to an image or dir of images to segment.')
parser.add_argument('out_dir', help='directory to save the results')
parser.add_argument('-m', '--masks-too', action='store_true',
					help='whether or not to save individual masks as images')

args = parser.parse_args()

inference_lib.run_inference(args.img_file_or_dir, args.out_dir)