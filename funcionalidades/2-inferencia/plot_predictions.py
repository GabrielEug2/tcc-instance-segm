import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Plot the predictions on the images for an easier interpretation')
parser.add_argument('pred_dir', help='Directory containing the predictions')
parser.add_argument('img_dir', help='Directory containing the images the predictions refer to')
parser.add_argument('out_dir', help=('Directory to save the results. Use the same directory '
                                    'you used to plot the annotations, so you can see them '
                                    'side by side.'))
args = parser.parse_args()

pred_dir = Path(args.pred_dir)
if not pred_dir.exists():
	raise FileNotFoundError(str(pred_dir))

img_dir = Path(args.img_dir)
if not img_dir.exists():
	raise FileNotFoundError(str(img_dir))

out_dir = Path(args.out_dir)
if out_dir.exists():
	op = input((f'out_dir "{str(out_dir)}" exists. Do you want '
	             'to overwrite it? [y/n] ')).lower()
	if op != 'y':
		print('Operation cancelled.')
		exit()

# Import depois pro --help ser rápido
from segm_lib.misc.plot_utils import plot_predictions
plot_predictions(pred_dir, img_dir, out_dir)