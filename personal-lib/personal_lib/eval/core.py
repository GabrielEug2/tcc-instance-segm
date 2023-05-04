from dataclasses import dataclass
import json
from pathlib import Path
import shutil

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from personal_lib.core import Annotations
from personal_lib.core import Predictions
from personal_lib.core import mask_conversions

@dataclass
class DatasetInfo:
	n_images: int
	n_objects: int
	class_dist: dict[str, int]

@dataclass
class ResultsPerModel:
	n_objects_predicted: int
	raw_pred_class_dist: dict[str, int]
	
	n_objects_considered_for_eval: int
	evaluated_pred_class_dist: dict[str, int]

	n_anns_considered_for_eval: int
	evaluated_anns_class_dist: dict[str, int]

	mAP: float

	n_true_positives: int
	n_false_positives: int
	n_false_negatives: int

@dataclass
class ResultsPerModelPerImage:
	n_objects_predicted: int
	raw_pred_class_dist: dict[str, int]
	
	n_objects_considered_for_eval: int
	evaluated_pred_class_dist: dict[str, int]

	n_anns_considered_for_eval: int
	evaluated_anns_class_dist: dict[str, int]

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
	per_model: dict[str, ResultsPerModel]
	per_img: dict[str, ResultsPerImage]

def evaluate_all(pred_dir: Path, ann_file: Path, out_dir: Path):
	ann_manager = Annotations(ann_file)
	ann_manager.normalize_classnames()
	ann_manager.save(ann_file)

	classmap = ann_manager.classmap()
	img_map = ann_manager.img_map()

	n_images = len(img_map)
	dataset_class_dist = ann_manager.class_distribution()
	n_objects = 0
	for count in dataset_class_dist.values():
		n_objects += count
	dataset_info = DatasetInfo(n_images, n_objects, dataset_class_dist)
	_save_dataset_info(dataset_info, out_dir)

	results = EvalResults({}, {})
	pred_manager = Predictions(pred_dir)
	model_names = pred_manager.get_model_names()
	for model_name in model_names:
		pred_manager.copy_to(model_name, out_dir / model_name / 'raw_predictions')

		raw_pred_class_dist = pred_manager.class_distribution(model_name)
		n_objects_predicted = 0
		for count in raw_pred_class_dist.values():
			n_objects_predicted += count

		_filter_common_classes(ann_manager, pred_manager, out_dir / model_name)

		evaluatable_classes = _filter_common_classes('groundtruth', model_name, class_dist_out_dir)
		model_eval_out_dir = out_dir / 'tmp' / model_name

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

		results.per_model[model_name] = ResultsPerModel(
			n_objects_predicted,
			raw_pred_class_dist,


		)

		results[model_name] = evaluate_model(ground_truth, detections, img_map)



	_save_results(results, out_dir)
	shutil.rmtree(out_dir / 'tmp')

def _save_dataset_info(dataset_info: DatasetInfo, out_dir: Path):
	dataset_info_file = out_dir / 'dataset-info.json'
	with dataset_info_file.open('w') as f:
		json.dump(dataset_info, f)

def _filter_common_classes(ann_manager: Annotations, pred_manager: Predictions, out_dir: Path):
	classes_in_anns = ann_manager.class_list()
	classes_in_preds = pred_manager.class_list(model_name)

	with Path(class_dist_out_dir / f"{code_a}.json").open('r') as f:
		dist_a = json.load(f)

	with Path(class_dist_out_dir / f"{code_b}.json").open('r') as f:
		dist_b = json.load(f)

	class_list_a = dist_a.keys()
	class_list_b = dist_b.keys()

	common_classes = [n for n in class_list_a if n in class_list_b]

	return common_classes

def _prepare_anns(ann_file: Path, evaluatable_classes: list, out_dir: Path):
	ann_manager = Annotations(ann_file)
	filtered_anns = ann_manager.filter_by_classes(evaluatable_classes)

	filtered_anns_file = out_dir / 'annotations_used_for_eval.json'
	filtered_anns_file.parent.mkdir(parents=True, exist_ok=True)
	with filtered_anns_file.open('w') as f:
		json.dump(filtered_anns, f)

	return filtered_anns_file

def _prepare_preds(
		pred_manager: Predictions,
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