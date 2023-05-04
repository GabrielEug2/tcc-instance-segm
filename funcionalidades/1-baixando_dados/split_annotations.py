import argparse
import json
from pathlib import Path
from personal_lib.coco_ann_parser import COCO_AnnParser

parser = argparse.ArgumentParser()
parser.add_argument('ann_file', help='file containing the annotations')
parser.add_argument('out_dir', help='directory to save the results')
args = parser.parse_args()

ann_file = Path(args.ann_file)
out_dir = Path(args.out_dir)

ann_parser = COCO_AnnParser(ann_file)

classmap = ann_parser.classmap()
img_map = ann_parser.img_map()

out_dir.mkdir(parents=True)
for img_name, id in img_map.items():
	anns_for_img = ann_parser.filter_by_img_id(id)

	# TODO continuar daqui
	anns_for_img.to_custom_format

	out_file = out_dir / img_name
	with out_file.open('w') as f:
		json.dump(f)