import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from personal_lib.parsing.annotations import Annotations
from personal_lib.parsing.common import mask_conversions
from personal_lib.parsing.predictions import get_model_names, PredictionManager

def evaluate_all(ann_file: Path, pred_dir: Path):
	# Adapted from: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

	annotations = Annotations(ann_file)
	img_map = annotations.img_map
	classmap = annotations.classmap

	ground_truth = COCO(ann_file)

	model_names = get_model_names(pred_dir)
	for model_name in model_names:
		pred_files = pred_dir.glob(f"*_{model_name}.json")

		coco_formatted_pred_file = pred_dir / f'coco-format_{model_name}.json'
		preds_to_coco_format(pred_files, img_map, classmap, coco_formatted_pred_file)

		detections = COCO.loadRes(coco_formatted_pred_file)
		evaluate_model(ground_truth, detections)

# def filter_common_classes():
# 	"""Filter classes that exist on both datasets (COCO and Openimages).

# 	Returns:
# 		list: list of class names, sorted by n_ocurrences on COCO.
# 	"""
# 	with COCO_CLASS_DIST_FILE.open('r') as f:
# 		coco_class_dist = json.load(f)

# 	import fiftyone.utils.openimages as openimages

# 	coco_classes = coco_class_dist.keys()
# 	openimages_classes = [x.lower() for x in openimages.get_segmentation_classes()]
# 	common_classes = [x for x in coco_classes if x in openimages_classes]

# 	filtered_dist = { name: count for name, count in coco_class_dist.items() if name in common_classes }
# 	sorted_class_counts = sorted(filtered_dist.items(), key=lambda c: c[1], reverse=True)
# 	sorted_classes = [c[0] for c in sorted_class_counts]
	
# 	return sorted_classes

def preds_to_coco_format(pred_files, img_map, classmap, out_file):
	coco_formatted_data = []
	for pred_file in pred_files:
		img_file = pred_file.name.split('_')[0]
		img_id = img_map[img_file]
		predictions = PredictionManager(pred_file)

		for pred in predictions:
			formatted_pred = {
				"image_id": img_id,
				"category_id": classmap[pred['classname']],
				"segmentation": mask_conversions.bin_mask_to_rle(pred['mask']),
				"score": pred['confidence']
			}
			coco_formatted_data.append(formatted_pred)

	with out_file.open('w') as f:
		json.dump(coco_formatted_data, f)

def evaluate_model(ground_truth, detections):
	# The usage for CocoEval is as follows:
	#  cocoGt=..., cocoDt=...       # load dataset and results
	#  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
	#  E.params.recThrs = ...;      # set parameters as desired
	#  E.evaluate();                # run per image evaluation
	#  E.accumulate();              # accumulate per image results
	#  E.summarize();               # display summary metrics of results
	# For example usage see evalDemo.m and http://mscoco.org/.

	E = COCOeval(ground_truth, detections)

	pass