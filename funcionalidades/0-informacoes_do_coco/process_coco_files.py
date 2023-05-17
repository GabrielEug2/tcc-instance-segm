import argparse
from pathlib import Path

# My path to just copy-paste when needed
# /mnt/e/Desktop/TCC/Dados/COCO/annotations/
parser = argparse.ArgumentParser(description='Parse useful information from the COCO dataset.')
parser.add_argument('coco_ann_dir', help='Directory where you placed the COCO annotations')
args = parser.parse_args()

coco_ann_dir = Path(args.coco_ann_dir)
if not coco_ann_dir.exists():
    raise FileNotFoundError(str(coco_ann_dir))

CLASS_DIST_OUT_FILE = Path(__file__).parent / 'coco_classdist.json'
CLASS_MAP_OUT_FILE = Path(__file__).parent / 'coco_classmap.json'


# Late import just to make sure --help is fast
from segm_lib.misc import coco_utils

print('Class dist... ', end='')
coco_utils.class_dist(coco_ann_dir, CLASS_DIST_OUT_FILE, verbose=True)
print('done')

print('Class map... ', end='')
coco_utils.class_map(coco_ann_dir, CLASS_MAP_OUT_FILE)
print('done')