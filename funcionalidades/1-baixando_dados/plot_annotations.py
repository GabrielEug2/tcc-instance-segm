import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Plot the annotations on the images for an easier interpretation')
parser.add_argument('ann_dir', help='Directory containing the annotations')
parser.add_argument('img_dir', help='Directory containing the images the annotations refer to')
parser.add_argument('out_dir', help='Directory to save the results')
args = parser.parse_args()

ann_dir = Path(args.ann_dir)
if not ann_dir.exists():
	raise FileNotFoundError(str(ann_dir))

img_dir = Path(args.img_dir)
if not img_dir.exists():
	raise FileNotFoundError(str(img_dir))

out_dir = Path(args.out_dir)
if out_dir.exists():
	op = input((f'out_dir "{str(out_dir)}" exists. Do you want '
	             'to overwrite it? [y/n] ')).strip().lower()
	if op != 'y':
		print('Operation cancelled.')
		exit()

# Import depois pro --help ser rápido
from segm_lib.plot.detectron_plot_lib import DetectronPlotLib
from segm_lib.plot.bin_plot_lib import BinPlotLib

print('Pretty plot...')
plot_lib = DetectronPlotLib()
plot_lib.plot_annotations(ann_dir, img_dir, out_dir)

print('\nBin plot...')
plot_lib = BinPlotLib()
plot_lib.plot_annotations(ann_dir, img_dir, out_dir)