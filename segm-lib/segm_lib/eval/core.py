from dataclasses import dataclass
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from segm_lib.annotations import Annotations
from segm_lib.predictions import Predictions
from segm_lib.coco_annotations import COCOAnnotations

@dataclass
class DatasetInfo:
	n_images: int
	n_objects: int
	class_dist: dict[str, int]

@dataclass
class RawResults:
	n_images_with_predictions: int = 0
	n_objects_predicted: int = 0
	class_dist: dict[str, int] = {}

@dataclass
class EvalFilters:
	classes_considered: list[str] = []
	pred_classes_ignored: list[str] = []
	ann_classes_ignored: list[str] = []
	n_preds_considered: int = 0
	n_anns_considered: int = 0

@dataclass
class DatasetResults:
	mAP: float = 0.0
	n_true_positives: int = 0
	n_false_positives: int = 0
	n_false_negatives: int = 0

@dataclass
class ImageResults:
	n_objects_annotated: int
	ann_class_dist: dict[str, int]

	n_objects_predicted: int
	pred_class_dist: dict[str, int]
	
	pred_classes_ignored: list[str]
	ann_classes_ignored: list[str]
	n_preds_considered: int
	n_anns_considered: int

	mAP: float
	true_positives: list # preds
	false_positives: list # preds
	false_negatives: list # preds
	
@dataclass
class ModelResults:
	raw_results: RawResults = None
	eval_filters: EvalFilters = None
	results_on_dataset: DatasetResults = None
	results_per_image: dict[str, ImageResults] = {}


def evaluate_all(pred_dir: Path, ann_dir: Path, ann_file: Path, out_dir: Path):
	ann_manager = Annotations(ann_dir)
	dataset_info = _compute_dataset_info(ann_manager)
	_save_dataset_info(dataset_info, out_dir)

	pred_manager = Predictions(pred_dir)
	model_names = pred_manager.get_model_names()
	for model_name in model_names:
		eval_dir = out_dir / model_name
		results = _evaluate_model(model_name, pred_manager, ann_manager, ann_file, eval_dir)
		_save_results(results, eval_dir)

	# Save it per model, then parse it to per image

def _compute_dataset_info(ann_manager: Annotations) -> DatasetInfo:
	n_images = ann_manager.get_n_images()
	n_objects = ann_manager.get_n_objects()
	class_dist = ann_manager.class_distribution()

	return DatasetInfo(n_images, n_objects, class_dist)

def _save_dataset_info(dataset_info: DatasetInfo, out_dir: Path):
	dataset_info_file = out_dir / 'dataset-info.json'
	with dataset_info_file.open('w') as f:
		json.dump(dataset_info, f)

def _evaluate_model(model_name: str, pred_manager: Predictions, ann_manager: Annotations, ann_file: Path, out_dir: Path) -> ModelResults:
	results = ModelResults()

	results.raw_results = _compute_raw_results(pred_manager, model_name)

	ann_classes = ann_manager.get_classnames()
	pred_classes = pred_manager.get_classnames(model_name)
	results.eval_filters = _compute_eval_filters(ann_classes, pred_classes)

	common_classes = results.eval_filters.classes_considered

	filtered_ann_dir = out_dir / 'annotations_used'
	filtered_pred_dir = out_dir / 'predictions_used'
	ann_manager.filter(common_classes, filtered_ann_dir)
	pred_manager.filter(model_name, common_classes, filtered_pred_dir)
	# I would use my own annotation format, but the eval API uses some
	# fields I don't save on mine, like area. It's easier to filter the
	# original file too than modifying my format to include all those fields.
	filtered_ann_file = out_dir / 'internal' / 'annotations.json'
	COCOAnnotations(ann_file).filter(common_classes, filtered_ann_file)

	filtered_ann_manager = Annotations(filtered_ann_dir)
	filtered_pred_manager = Predictions(filtered_pred_dir)
	results.eval_filters.n_anns_considered = filtered_ann_manager.get_n_objects()
	results.eval_filters.n_preds_considered = filtered_pred_manager.get_n_objects()

	results.results_on_dataset = _evaluate_on_dataset(model_name, filtered_pred_manager, filtered_ann_manager, filtered_ann_file, out_dir)
	# results.results_per_image = _evaluate_per_image()

	return results

def _compute_raw_results(pred_manager: Predictions, model_name: str) -> RawResults:
	n_images_with_predictions = pred_manager.get_n_images_with_predictions(model_name)
	n_objects_predicted = pred_manager.get_n_objects(model_name)
	pred_class_dist = pred_manager.class_distribution(model_name)

	return RawResults(
		n_images_with_predictions=n_images_with_predictions,
		n_objects_predicted=n_objects_predicted,
		class_dist=pred_class_dist,
	)

def _compute_eval_filters(ann_classes: list[str], pred_classes: list[str]) -> EvalFilters:
	common_classes = _filter_common(ann_classes, pred_classes)
	pred_classes_ignored = pred_classes - common_classes
	ann_classes_ignored = ann_classes - common_classes

	return EvalFilters(
		classes_considered=common_classes,
		pred_classes_ignored=pred_classes_ignored,
		ann_classes_ignored=ann_classes_ignored,
		n_preds_considered=0, # vai ser computado depois
		n_anns_considered=0, # vai ser computado depois
	)

def _filter_common(list_a: list[str], list_b: list[str]):
	return [n for n in list_a if n in list_b]

def _evaluate_on_dataset(
		model_name: str,
		filtered_pred_manager: Predictions,
		filtered_ann_manager: Annotations,
		filtered_ann_file: Path,
		out_dir: Path
	):
	ann_file_for_eval = filtered_ann_file
	filtered_coco_ann_manager = COCOAnnotations(filtered_ann_file)
	classmap = filtered_coco_ann_manager.classmap()	
	img_map = filtered_coco_ann_manager.img_map()

	pred_file_for_eval = out_dir / 'internal' / 'predictions.json'
	filtered_pred_manager.to_coco_format(model_name, img_map, classmap, pred_file_for_eval)

	ground_truth = COCO(ann_file_for_eval)
	detections = ground_truth.loadRes(str(pred_file_for_eval))

	E = COCOeval(ground_truth, detections)
	E.evaluate()
	E.accumulate()
	E.summarize()
	mAP_on_all_images = round(E.stats[0], 3)

	import pdb; pdb.set_trace()
	n_true_positives = 0
	n_false_positives = 0
	n_false_negatives = 0

	return DatasetResults(
		mAP=mAP_on_all_images,
		n_true_positives=n_true_positives,
		n_false_positives=n_false_positives,
		n_false_negatives=n_false_negatives,
	)

def _evaluate_per_image(
		model_name: str,
		filtered_pred_manager: Predictions,
		filtered_ann_manager: Annotations,
		filtered_ann_file: Path,
		out_dir: Path
	):
	filtered_coco_ann_manager = COCOAnnotations(filtered_ann_file)
	classmap = filtered_coco_ann_manager.classmap()
	img_map = filtered_coco_ann_manager.img_map()

	pred_file_for_eval = out_dir / 'internal' / 'predictions.json'

	results_per_img = {}
	for img_file_name, img_id in img_map.items():
		pred_dir_for_view = None # some dir
		filtered_ann_manager.filter(img_filename=img_file_name, out_dir=ann_dir_for_view)
		ann_dir_for_view = None # some dir
		filtered_pred_manager.filter(img_filename=img_file_name, out_dir=pred_dir_for_view)

		ann_file_for_eval = None # some temp file
		filtered_coco_ann_manager.filter_by_img_id(img_id, out_file=ann_file_for_eval)
		pred_file_for_eval = None # some temp file
		pred_dir_filtered_by_img_manager = Predictions(pred_dir_for_view)
		pred_dir_filtered_by_img_manager.to_coco_format(model_name, img_map, classmap, pred_file_for_eval)

		ground_truth = COCO(ann_file_for_eval)
		detections = ground_truth.loadRes(str(pred_file_for_eval))

		E = COCOeval(ground_truth, detections)
		E.evaluate()
		E.accumulate()
		E.summarize()

		image_results = ImageResults()

		mAP = round(E.stats[0], 3)
		import pdb; pdb.set_trace()
		true_positives = []
		false_positives = []
		false_negatives = []

		results_per_img[img_file_name] = image_results

	return results_per_img

def _save_results(results: ModelResults, out_dir: Path):
	pass