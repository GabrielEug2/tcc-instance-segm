import argparse

import inference_lib

parser = argparse.ArgumentParser(
    description='Runs instance segmentation on a set of images and store '
                'the results in the specified folder'
)
parser.add_argument('img_dir', help='directory containing the images to segment')
parser.add_argument('output_dir', help='directory to save the results')
parser.add_argument('-m', '--save_masks', help='whether or not to save individual masks as images')

args = parser.parse_args()

inference_lib.run_on_all_models(args.img_dir, args.output_dir, args.save_masks)