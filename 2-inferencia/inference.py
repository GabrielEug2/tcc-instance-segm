import argparse

import inference_lib

parser = argparse.ArgumentParser(
    description='Runs instance segmentation on a set of images and save '
                'the results in the specified folder'
)
parser.add_argument('img_dir', help='directory containing the images to segment')
parser.add_argument('out_dir', help='directory to save the results')

args = parser.parse_args()

inference_lib.run_on_all_models(args.img_dir, args.out_dir)