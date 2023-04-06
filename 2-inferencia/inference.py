import argparse

import inference_lib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs instance segmentation on a set of images and save '
                    'the results in the specified folder'
    )
    parser.add_argument('img_dir', help='directory containing the images to segment')
    parser.add_argument('out_dir', help='directory to save the results')

    args = parser.parse_args()

    inference_lib.inference(args.img_dir, args.out_dir)