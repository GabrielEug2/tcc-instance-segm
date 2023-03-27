import argparse

import view_lib

parser = argparse.ArgumentParser()
parser.add_argument('img_file', help='path to the image')
parser.add_argument('pred_dir', help='dir containing the predictions for said image')
parser.add_argument('out_dir', help='directory to save the results')
parser.add_argument('-m', '--masks-too', help='whether or not to save individual masks as images',
                    action='store_true')

args = parser.parse_args()

view_lib.plot_annotations(args.img_file, args.ann_file, args.out_dir, args.save_masks)