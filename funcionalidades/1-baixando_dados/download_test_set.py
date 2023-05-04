
import argparse
import json
import random
from pathlib import Path

# Os imports do fiftyone não estão aqui porque é muito lento.
# Se deixar no topo da arquivo, o --help também demora

parser = argparse.ArgumentParser()
parser.add_argument('n_imgs', type=int, help='Number of images to download')
parser.add_argument('out_dir', help='Directory to save')
args = parser.parse_args()

n_imgs = args.n_imgs
out_dir = Path(args.out_dir)


import fiftyone.utils.openimages as openimages
import fiftyone.zoo as fozoo
from fiftyone.types import COCODetectionDataset

COCO_CLASS_DIST_FILE = Path(__file__).parent / 'coco_classdist.json'
with COCO_CLASS_DIST_FILE.open('r') as f:
	coco_class_dist = json.load(f)

coco_classes = [c.lower() for c in coco_class_dist]
openimages_classes = [c.lower() for c in openimages.get_segmentation_classes()]
common_classes = [c for c in coco_classes if c in openimages_classes]

filtered_dist = { n: c for n, c in coco_class_dist.items() if n in common_classes }
dist_sorted_by_count = sorted(filtered_dist.items(), key=lambda c: c[1], reverse=True)
common_classes_sorted_by_count = [c[0] for c in dist_sorted_by_count]

top_10_classes = common_classes_sorted_by_count[:10]

top_10_classes = [x.capitalize() for x in top_10_classes]
dataset = fozoo.load_zoo_dataset(
	"open-images-v6",
	split="validation",
	label_types="segmentations",
	max_samples=n_imgs,
	classes=top_10_classes,
	shuffle=True,
	seed=random.randrange(0, 1000),
)
dataset.export(
	export_dir=str(out_dir),
	dataset_type=COCODetectionDataset
)
Path.rename(out_dir / 'data', out_dir / 'images')
Path.rename(out_dir / 'labels.json', out_dir / 'annotations.json')