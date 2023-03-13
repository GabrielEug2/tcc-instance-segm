
import argparse

import inference_lib.core

def build_parser():
    parser = argparse.ArgumentParser(
        description='Runs instance segmentation on a set of images and store '
                    'the results in the specified folder'
    )
    parser.add_argument('input_dir', help='directory containing the images to segment')
    parser.add_argument('output_dir', help='directory to save the results')
    return parser

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    args.input_dir = args.input_dir.rstrip('/')
    args.output_dir = args.input_dir.rstrip('/')

    inference_lib.core.run_on_all_models(args.input_dir, args.output_dir)