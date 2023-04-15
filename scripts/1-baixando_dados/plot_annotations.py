import argparse
from pathlib import Path

import plot_lib

parser = argparse.ArgumentParser(description='Plot the annotations on the images for an easier interpretation')
parser.add_argument('ann_dir', help='directory containing images and annotations for said image(s).')
parser.add_argument('out_dir', help='directory to save the results')
parser.add_argument('-m', '--save-masks', action='store_true',
					help='whether or not to save individual masks as images')

args = parser.parse_args()

ann_dir = Path(args.ann_dir)
out_dir = Path(args.out_dir)
plot_lib.annotations.plot(ann_dir, out_dir, args.save_masks)