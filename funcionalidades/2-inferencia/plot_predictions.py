import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Plot the predictions on the images for an easier interpretation')
parser.add_argument('pred_dir', help='directory containing the predictions')
parser.add_argument('img_dir', help='directory containing the images the predictions refer to')
parser.add_argument('out_dir', help=('directory to save the results. Use the same directory you used '
                                    'to plot the annotations, so you can see them side by side.'))
args = parser.parse_args()

pred_dir = Path(args.pred_dir)
img_dir = Path(args.img_dir)
out_dir = Path(args.out_dir)

# Import depois pro --help ser rápido
from personal_lib.plot import plot_predictions
plot_predictions(pred_dir, img_dir, out_dir)