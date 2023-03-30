import argparse

import view_lib

def plot_annotations(args):
    view_lib.plot_annotations(args.img_file_or_dir, args.ann_file, args.out_dir)

def plot_predictions(args):
    view_lib.plot_predictions(args.img_file_or_dir, args.pred_dir, args.out_dir)

def plot_both(args):
    view_lib.plot_annotations(args.img_file_or_dir, args.ann_file, args.out_dir)
    view_lib.plot_predictions(args.img_file_or_dir, args.pred_dir, args.out_dir)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--masks-too', help='whether or not to save individual masks as images',
                         action='store_true')
subparsers = parser.add_subparsers()

parser_ann = subparsers.add_parser('anns', help='plot annotations')
parser_ann.add_argument('img_file_or_dir', help='image or directory of images')
parser_ann.add_argument('ann_file', help='file containing the annotations for said image(s).')
parser_ann.add_argument('out_dir', help='directory to save the results')
parser_ann.set_defaults(func=plot_predictions)

parser_pred = subparsers.add_parser('pred', help='plot predictions')
parser_pred.add_argument('img_file_or_dir', help='image or directory of images')
parser_pred.add_argument('pred_dir', help='directory containing the predictions for said image(s)')
parser_pred.add_argument('out_dir', help='directory to save the results')
parser_pred.set_defaults(func=plot_predictions)

parser_both = subparsers.add_parser('both', help='plot both annotations and predictions')
parser_both.add_argument('img_file_or_dir', help='image or directory of images')
parser_both.add_argument('ann_file', help='file containing the annotations for said image(s).')
parser_both.add_argument('pred_dir', help='directory containing the predictions for said image(s)')
parser_both.add_argument('out_dir', help='directory to save the results')
parser_ann.set_defaults(func=plot_both)

args = parser.parse_args()
args.func(args)