from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import shutil

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from segm_lib.annotations import Annotations
from segm_lib.predictions import Predictions
from segm_lib import mask_conversions

@dataclass
class DatasetInfo:
	n_images: int
	n_objects: int
	class_dist: dict[str, int]

@dataclass
class ResultsPerModel:
	n_images_with_predictions: int = 0
	n_objects_predicted: int = 0
	class_dist: dict[str, int] = {}
	
	classes_considered_on_eval: list[str] = []
	pred_classes_ignored: list[str] = []
	ann_classes_ignored: list[str] = []
	n_objects_considered_for_eval: int = 0
	n_anns_considered_for_eval: int = 0

	mAP: float = 0.0
	n_true_positives: int = 0
	n_false_positives: int = 0
	n_false_negatives: int = 0

@dataclass
class ResultsPerModelPerImage:
	n_objects_predicted: int
	class_dist: dict[str, int]
	
	n_objects_considered_for_eval: int
	n_anns_considered_for_eval: int
	pred_classes_ignored_on_eval: list[str]
	ann_classes_ignored_on_eval: list[str]

	mAP: float
	true_positives: list # preds
	false_positives: list # preds
	false_negatives: list # preds

@dataclass
class ResultsPerImage:
	n_objects: int
	class_dist: dict[str, int]
	per_model: dict[str, ResultsPerModelPerImage]

@dataclass
class EvalResults:
	results_per_model: dict[str, ResultsPerModel] = defaultdict(None)
	results_per_img: dict[str, ResultsPerImage] = defaultdict(None)

def evaluate_all(pred_dir: Path, ann_dir: Path, out_dir: Path):
	ann_manager = Annotations(ann_dir)
	n_images = ann_manager.get_n_images()
	dataset_class_dist = ann_manager.class_distribution()
	n_objects = ann_manager.get_n_objects()
	ann_classes = dataset_class_dist.keys()

	dataset_info = DatasetInfo(n_images, n_objects, dataset_class_dist)
	_save_dataset_info(dataset_info, out_dir)

	pred_manager = Predictions(pred_dir)
	model_names = pred_manager.get_model_names()
	for model_name in model_names:
		eval_dir = out_dir / model_name
		results = ResultsPerModel()

		n_images_with_predictions = pred_manager.get_n_images_with_predictions()
		pred_class_dist = pred_manager.class_distribution(model_name)
		n_objects_predicted = pred_manager.get_n_objects(model_name)
		results.n_images_with_predictions = n_images_with_predictions
		results.n_objects_predicted = n_objects_predicted
		results.class_dist = pred_class_dist

		pred_classes = pred_class_dist.keys()
		common_classes = _filter_common(ann_classes, pred_classes)
		results.classes_considered_on_eval = common_classes
		results.ann_classes_ignored = ann_classes - common_classes
		results.pred_classes_ignored = pred_classes - common_classes

		filtered_ann_dir = eval_dir / 'annotations_used'
		filtered_pred_dir = eval_dir / 'predictions_used'
		ann_manager.copy_to(filtered_ann_dir, common_classes)
		pred_manager.copy_to(filtered_pred_dir, model_name, common_classes)

		filtered_ann_manager = Annotations(filtered_ann_dir)
		filtered_pred_manager = Predictions(filtered_pred_dir)
		results.n_anns_considered_for_eval = filtered_ann_manager.get_n_objects()
		results.n_objects_considered_for_eval = filtered_pred_manager.get_n_objects(model_name)

		ann_file_for_eval = _prepare_anns(filtered_ann_manager, eval_dir)
		pred_file_for_eval = _prepare_preds(filtered_pred_manager, model_name, common_classes)
		ground_truth = COCO(ann_file_for_eval)
		detections = ground_truth.loadRes(str(pred_file_for_eval))

		# @dataclass
		# class ResultsPerModel:
		# 	mAP: float
		# 	n_true_positives: int
		# 	n_false_positives: int
		# 	n_false_negatives: int
		results[model_name] = evaluate_model(ground_truth, detections, img_map)
		# TODO how to fill results_per_img

	results = EvalResults({}, {})


	_save_results(results, out_dir)
	shutil.rmtree(out_dir / 'tmp')

def _save_dataset_info(dataset_info: DatasetInfo, out_dir: Path):
	dataset_info_file = out_dir / 'dataset-info.json'
	with dataset_info_file.open('w') as f:
		json.dump(dataset_info, f)

def _filter_common(list_a: list[str], list_b: list[str]):
	return [n for n in list_a if n in list_b]

def _prepare_anns(ann_manager: Annotations, eval_dir: Path):
	ann_file_for_eval = eval_dir / 'eval_files' / 'coco-annotations.json'
	ann_file_for_eval.parent.mkdir(parents=True, exist_ok=True)

	ann_manager.to_coco_format(ann_file_for_eval)

	return ann_file_for_eval

def _prepare_preds(pred_manager: Predictions, model_name: str, eval_dir: Path):
	pred_file_for_eval = eval_dir / 'eval_files' / 'coco-annotations.json'
	pred_file_for_eval.parent.mkdir(parents=True, exist_ok=True)

	pred_manager.to_coco_format(pred_file_for_eval, model_name)

	return pred_file_for_eval

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
		import pdb; pdb.set_trace()
		# E.evalImgs.
		mAP_per_image[img_file_name] = round(E.stats[0], 3)

	return EvalResults(mAP_on_all_images, mAP_per_image)

def _save_results(results_per_model: dict[str, EvalResults], out_dir: Path):
	result_dict = {
		'mAP_on_all_images': {},
		'mAP_per_image': {},
	}
	for model_name, results in results_per_model.items():
		result_dict['mAP_on_all_images'][model_name] = results.ap_on_all_imgs
		
		for img, ap_for_image in results.ap_per_image.items():
			if img not in result_dict['mAP_per_image']:
				result_dict['mAP_per_image'][img] = {}

			result_dict['mAP_per_image'][img][model_name] = ap_for_image

	results_file = out_dir / 'results.json'
	with results_file.open('w') as f:
		json.dump(result_dict, f)