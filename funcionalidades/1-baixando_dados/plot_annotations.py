import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Plot the annotations on the images for an easier interpretation')
parser.add_argument('ann_file', help='file containing the annotations')
parser.add_argument('img_dir', help='directory containing the images the annotations refer to')
parser.add_argument('out_dir', help='directory to save the results')
args = parser.parse_args()

ann_file = Path(args.ann_file)
img_dir = Path(args.img_dir)
out_dir = Path(args.out_dir)

# Import depois pro --help ser rápido
from personal_lib.plot import plot_annotations
plot_annotations(ann_file, img_dir, out_dir)