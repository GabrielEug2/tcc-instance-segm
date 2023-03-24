import argparse
from pathlib import Path

SAMPLE_DIR = Path(__file__).parent / 'sample'


parser = argparse.ArgumentParser()
parser.add_argument('img_dir', help='directory containing the images',
                    nargs='?', default=str(SAMPLE_DIR / 'images'))

parser.add_argument('ann_file', help='file containing the annotations for the images',
                    nargs='?', default=str(SAMPLE_DIR / 'annotations.json'))

parser.add_argument('pred_dir', help='directory containing the predictions',
                    nargs='?', default=str(SAMPLE_DIR / 'predictions'))

args = parser.parse_args()

evaluate(args.img_dir, args.ann_file, args.pred_dir)