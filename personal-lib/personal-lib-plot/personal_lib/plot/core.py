from pathlib import Path

from personal_lib.core import mask_conversions

from . import common_logic

def plot_annotations(dataset_dir: Path, out_dir: Path, save_masks: bool):
	from personal_lib import ann_logic

	ann_file = dataset_dir / 'annotations.json'
	img_dir = dataset_dir / 'images'
	img_files = img_dir.glob('*.jpg')

	anns = ann_logic.load_anns(ann_file)
	classmap = ann_logic.extract_classmap(anns)

	if not out_dir.exists():
		out_dir.mkdir()

	for img_file in img_files:
		relevant_anns = ann_logic.get_anns_for_img(img_file, anns)
		if len(relevant_anns) == 0:
			print(f"No annotations found on \"{str(ann_file)} for image \"{str(img_file)}\". Skipping")
			continue

		h, w = ann_logic.get_img_dimensions(img_file, anns)
		formatted_anns = _to_plot_format(relevant_anns, h, w, classmap)
		
		annotated_img_file = out_dir / f"{img_file.stem}_groundtruth.jpg"
		common_logic.plot(formatted_anns, img_file, annotated_img_file)

		if save_masks:
			mask_out_dir = out_dir / f"{img_file.stem}_groundtruth_masks"
			common_logic.plot_individual_masks(formatted_anns, mask_out_dir, img_file)

def _to_plot_format(anns, img_h, img_w, classmap):
	formatted_anns = []

	for ann in anns:
		class_id = ann['category_id']
		classname = classmap[str(class_id)]
		confidence = 100.0

		mask = mask_conversions.ann_to_bin_mask(ann['segmentation'], img_h, img_w)

		bbox = ann['bbox'] # tá em [x1,y1,h,w], precisa estar em [x1,y1,x2,y2]
		x1, y1, h, w = bbox[0], bbox[1], bbox[2], bbox[3]
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

def plot_predictions(pred_dir: Path, img_files: list[Path], save_masks: bool):
	from personal_lib import pred_logic

	for img_file in img_files:
		pred_files = pred_dir.glob(f"{img_file.stem}_*.json")

		for pred_file in pred_files:
			model_name = pred_file.stem.split('_')[1]
			predictions = pred_logic.load_preds(pred_file)

			predictions_img_file = pred_dir / f"{img_file.stem}_{model_name}.jpg"
			common_logic.plot(predictions, img_file, predictions_img_file)

			if save_masks:
				mask_out_dir = pred_dir / f"{img_file.stem}_{model_name}_masks"
				common_logic.plot_individual_masks(predictions, mask_out_dir, img_file)