import argparse

import view_lib

def plot_predictions(args):
    # Call from inference_lib.core
    view_lib.plot_predictions(args.img_file_or_dir, args.pred_dir, args.out_dir)
