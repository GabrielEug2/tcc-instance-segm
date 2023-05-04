
from pathlib import Path

from tqdm import tqdm

from personal_lib.core import Annotations
from personal_lib.core import mask_conversions
from . import plot_lib

def plot(ann_file: Path, img_dir: Path, out_dir: Path):
	# TODO work with custom format annotations
	anns = Annotations(ann_file)
	classmap_by_id = anns.classmap_by_id()
	img_dimensions = anns.img_dimensions()
	img_files = img_dir.glob('*.jpg')

	for img_file in tqdm(img_files):
		relevant_anns = anns.filter_by_img(img_file.name)
		if len(relevant_anns) == 0:
			print(f"No annotations found on \"{str(ann_file)} for image \"{str(img_file)}\". Skipping")
			continue

		h, w = img_dimensions[img_file.name]
		formatted_anns = _to_plot_format(relevant_anns, h, w, classmap_by_id)
		
		annotated_img_file = out_dir / img_file.stem / "groundtruth.jpg"
		annotated_img_file.parent.mkdir(parents=True, exist_ok=True)
		plot_lib.plot(formatted_anns, img_file, annotated_img_file)

		mask_out_dir = out_dir / img_file.stem / "groundtruth_masks"
		mask_out_dir.mkdir(parents=True, exist_ok=True)
		plot_lib.plot_individual_masks(formatted_anns, mask_out_dir, img_file)

def _to_plot_format(anns, img_h, img_w, classmap_by_id):
	formatted_anns = []
	for ann in anns:
		cat_id = ann['category_id']
		classname = classmap_by_id[str(cat_id)]
		confidence = 100.0

		mask = mask_conversions.ann_to_bin_mask(ann['segmentation'], img_h, img_w)

		bbox = ann['bbox'] # tá em [x1,y1,w,h], precisa estar em [x1,y1,x2,y2]
		x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
		x2 = x1 + w
		y2 = y1 + h
		bbox = [x1, y1, x2, y2]

		pred = {
			'classname': classname,
			'confidence': confidence,
			'mask': mask,
			'bbox': bbox,
		}
		formatted_anns.append(pred)

	return formatted_anns