import argparse

import plot_lib

parser = argparse.ArgumentParser(
	description='Plot the annotations on the images'
)
parser.add_argument('img_file_or_dir', help='path to an image or dir of images.')
parser.add_argument('ann_file', help='file containing the annotations for said image(s).')
parser.add_argument('out_dir', help='directory to save the results')
parser.add_argument('-m', '--masks-too', action='store_true',
					help='whether or not to save individual masks as images')

args = parser.parse_args()

plot_lib.annotations.plot(args.img_file_or_dir, args.ann_file, args.out_dir)