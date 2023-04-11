import argparse

import plot_lib

parser = argparse.ArgumentParser(description='Plot the annotations on the images')
parser.add_argument('ann_dir', help='directory containing images and annotations for said image(s).')
parser.add_argument('out_dir', help='directory to save the results')
# parser.add_argument('-m', '--masks-too', action='store_true',
# 					help='whether or not to save individual masks as images')

args = parser.parse_args()

plot_lib.annotations.plot(args.ann_dir, args.out_dir)