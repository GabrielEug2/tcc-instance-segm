import argparse
import json
from pathlib import Path

from segm_lib.coco_ann_parser import COCOAnnParser
from segm_lib import mask_conversions

parser = argparse.ArgumentParser()
parser.add_argument('ann_file', help='file containing the annotations')
parser.add_argument('out_dir', help='directory to save the results')
args = parser.parse_args()

ann_file = Path(args.ann_file)
out_dir = Path(args.out_dir)

ann_parser = COCOAnnParser(ann_file)

classmap = ann_parser.classmap_by_id()
img_map = ann_parser.img_map()
img_dimensions = ann_parser.img_dimensions()

out_dir.mkdir(parents=True, exist_ok=True)
for img_name, id in img_map.items():
	anns_for_img = ann_parser.filter_by_img_id(id)

	custom_anns = []
	for ann in anns_for_img:
		classname = classmap[str(ann['category_id'])]

		img_h, img_w = img_dimensions[img_name]
		rle_mask = mask_conversions.ann_to_rle(ann['segmentation'], img_h, img_w)

		bbox = ann['bbox'] # Mesmo formato do COCO, [x, y, w, h]

		custom_anns.append({
			'classname': classname,
			'mask': rle_mask,
			'bbox': bbox,
		})

	out_file = out_dir / f"{img_name}.json"
	with out_file.open('w') as f:
		json.dump(custom_anns, f)