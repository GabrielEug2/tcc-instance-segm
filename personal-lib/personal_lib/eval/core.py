import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from personal_lib.parsing.annotations import AnnotationManager
from personal_lib.parsing.predictions import PredictionManager
from personal_lib.parsing.common import mask_conversions
from .class_dist import class_dist

def evaluate_all(ann_file: Path, pred_dir: Path, out_dir: Path):
	# Adapted from: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
	class_dist(ann_file, pred_dir, out_dir)

	ann_manager = AnnotationManager(ann_file)
	ann_manager.cat_to_lowercase()
	ann_manager.save(ann_file)
	ground_truth = COCO(ann_file)

	img_map = ann_manager.img_map()
	classmap = ann_manager.classmap()

	pred_manager = PredictionManager(pred_dir)
	model_names = pred_manager.get_model_names()
	for model_name in model_names:
		coco_formatted_pred_file = out_dir / 'coco-format-preds' / f'{model_name}.json'
		coco_formatted_pred_file.parent.mkdir(parents=True, exist_ok=True)
		_preds_to_coco_format(pred_manager, model_name, img_map, classmap, coco_formatted_pred_file)

		detections = ground_truth.loadRes(str(coco_formatted_pred_file))
		evaluate_model(ground_truth, detections)

def _preds_to_coco_format(pred_manager: PredictionManager, model_name: str, img_map: dict, classmap: dict, out_file: Path):
	coco_formatted_data = []
	for img_file in img_map:
		predictions = pred_manager.load(Path(img_file).stem, model_name)
		for pred in predictions:
			classname = pred['classname']
			if classname not in classmap:
				# can't evaluate if there are no annotations
				# of that class to compare
				continue

			formatted_pred = {
				"image_id": img_map[img_file],
				"category_id": classmap[classname],
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
	E.evaluate()
	E.accumulate()
	E.summarize()

	# TODO Extract what I want from E