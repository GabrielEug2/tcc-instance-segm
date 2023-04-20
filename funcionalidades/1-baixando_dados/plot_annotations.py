import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Plot the annotations on the images for an easier interpretation')
parser.add_argument('ann_dir', help='directory containing images and annotations for said image(s).')
parser.add_argument('out_dir', help='directory to save the results')
parser.add_argument('-m', '--save-masks', action='store_true',
					help='whether or not to save individual masks as images')

args = parser.parse_args()

ann_dir = Path(args.ann_dir)
out_dir = Path(args.out_dir)

ann_file = ann_dir / 'annotations.json'
img_dir = ann_dir / 'images'
img_files = img_dir.glob('*.jpg')

# Import depois pro --help ser rápido
from personal_lib.plot import plot_annotations
plot_annotations(ann_file, img_files, out_dir, args.save_masks)