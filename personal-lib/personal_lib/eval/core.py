from dataclasses import dataclass
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from personal_lib.parsing.annotations import AnnotationManager
from personal_lib.parsing.predictions import PredictionManager
from personal_lib.parsing.common import mask_conversions
from .class_dist import class_dist

@dataclass
class Results:
	model_name: str
	ap_on_all_imgs: float
	ap_per_image: dict[str, float]

def evaluate_all(pred_dir: Path, ann_file: Path, out_dir: Path):
	ann_manager = AnnotationManager(ann_file)
	ann_manager.normalize_classnames()
	ann_manager.save(ann_file)

	class_dist_out_dir = out_dir / 'class-dist'
	class_dist(pred_dir, ann_file, class_dist_out_dir)

	classmap = ann_manager.classmap()
	img_map = ann_manager.img_map()

	pred_manager = PredictionManager(pred_dir)
	model_names = pred_manager.get_model_names()
	results = []
	for model_name in model_names:
		evaluatable_classes = _filter_common_classes('groundtruth', model_name, class_dist_out_dir)
		model_eval_out_dir = out_dir / model_name

		ann_file_for_eval = _prepare_anns(ann_file, evaluatable_classes, model_eval_out_dir)
		ground_truth = COCO(ann_file_for_eval)

		pred_file_for_eval = _prepare_preds(
			pred_manager,
			model_name,
			evaluatable_classes,
			classmap,
			img_map,
			model_eval_out_dir
		)
		detections = ground_truth.loadRes(str(pred_file_for_eval))

		results_for_model = evaluate_model(ground_truth, detections, img_map)
		results_for_model.model_name = model_name
		results.append(results_for_model)

	print(results)

	# TODO Save on json that I can parse later to get best/worst images
	# _save_results()

def _filter_common_classes(code_a, code_b, class_dist_out_dir):
	with Path(class_dist_out_dir / f"{code_a}.json").open('r') as f:
		dist_a = json.load(f)

	with Path(class_dist_out_dir / f"{code_b}.json").open('r') as f:
		dist_b = json.load(f)

	class_list_a = dist_a.keys()
	class_list_b = dist_b.keys()

	common_classes = [n for n in class_list_a if n in class_list_b]

	return common_classes

def _prepare_anns(ann_file: Path, evaluatable_classes: list, out_dir: Path):
	ann_manager = AnnotationManager(ann_file)
	filtered_anns = ann_manager.filter_by_classes(evaluatable_classes)

	filtered_anns_file = out_dir / 'annotations_used_for_eval.json'
	filtered_anns_file.parent.mkdir(parents=True, exist_ok=True)
	with filtered_anns_file.open('w') as f:
		json.dump(filtered_anns, f)

	return filtered_anns_file

def _prepare_preds(
		pred_manager: PredictionManager,
		model_name: str,
		evaluatable_classes: list,
		classmap: dict,
		img_map: dict,
		out_dir: Path
	):
	coco_formatted_data = []
	for img_file_name in img_map:
		predictions = pred_manager.load(img_file_name, model_name)

		for pred in predictions:
			classname = pred['classname']
			if classname not in evaluatable_classes:
				# can't evaluate if there are no annotations
				# of that class to compare
				continue

			formatted_pred = {
				"image_id": img_map[img_file_name],
				"category_id": classmap[classname],
				"segmentation": mask_conversions.bin_mask_to_rle(pred['mask']),
				"score": pred['confidence']
			}
			coco_formatted_data.append(formatted_pred)

	coco_formatted_pred_file = out_dir / 'predictions_used_for_eval.json'
	coco_formatted_pred_file.parent.mkdir(parents=True, exist_ok=True)
	with coco_formatted_pred_file.open('w') as f:
		json.dump(coco_formatted_data, f)

	return coco_formatted_pred_file

def evaluate_model(ground_truth, detections, img_map):
	# Adapted from: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

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
	mAP_on_all_images = round(E.stats[0], 3)

	mAP_per_image = {}
	for img_file_name, img_id in img_map.items():
		print(img_id, img_file_name)
		E.params.imgIds = [img_id]
		E.evaluate()
		E.accumulate()
		E.summarize()
		mAP_per_image[img_file_name] = round(E.stats[0], 3)

	return Results(None, mAP_on_all_images, mAP_per_image)