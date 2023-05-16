from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from segm_lib.ann_manager import AnnManager
from segm_lib.pred_manager import PredManager
from segm_lib.eval.structures.dataset_info import DatasetInfo, ImageInfo
from segm_lib.eval.structures.model_results import RawResults, EvalFilters, ModelResults
from segm_lib.structures import Annotation, Prediction
from .actual_eval_code import prep_eval_files, evaluate_on_dataset, evaluate_per_image

def evaluate_all(pred_dir: Path, ann_dir: Path, ann_file: Path, out_dir: Path):
	ann_manager = AnnManager(ann_dir)
	dataset_info = _compute_dataset_info(ann_manager)
	_save_dataset_info(dataset_info, out_dir)

	pred_manager = PredManager(pred_dir)
	for model_name in pred_manager.get_model_names():
		eval_dir = out_dir / model_name
		eval_dir.mkdir(exist_ok=True)
		results = _evaluate_model(model_name, pred_manager, ann_manager, ann_file, eval_dir)
		_save_results(results, eval_dir)

	# Save it per model, then parse it to per image

def _compute_dataset_info(ann_manager: AnnManager) -> DatasetInfo:
	n_images = ann_manager.get_n_images()
	n_objects = ann_manager.get_n_objects()
	class_dist = ann_manager.class_distribution()

	info_per_img = {}
	for img in ann_manager.get_img_file_names():
		class_dist_on_img = ann_manager.class_dist_on_img(img)
		n_objects_on_img = sum(class_dist_on_img.values())

		info_per_img[img] = ImageInfo(
			n_objects=n_objects_on_img,
			class_dist=class_dist_on_img
		)

	return DatasetInfo(
		n_images=n_images,
		n_objects=n_objects,
		class_dist=class_dist,
		info_per_image=info_per_img
	)

def _save_dataset_info(dataset_info: DatasetInfo, out_dir: Path):
	dataset_info_dict = asdict(dataset_info)
	
	dataset_info_file = out_dir / 'dataset-info.json'
	dataset_info_file.parent.mkdir(parents=True, exist_ok=True)
	with dataset_info_file.open('w') as f:
		json.dump(dataset_info_dict, f, indent=4)

def _evaluate_model(
	model_name: str,
	pred_manager: PredManager,
	ann_manager: AnnManager,
	ann_file: Path,
	out_dir: Path
) -> ModelResults:
	print(f"\n\nEvaluating model '{model_name}'... ")
	results = ModelResults()
	results.raw_results = _compute_raw_results(model_name, pred_manager)

	ann_classes = ann_manager.get_classnames()
	pred_classes = pred_manager.get_classnames(model_name)
	results.eval_filters = _compute_eval_filters(ann_classes, pred_classes)

	common_classes = results.eval_filters.classes_considered
	eval_files = prep_eval_files(model_name, pred_manager, ann_manager, ann_file, common_classes, out_dir / 'eval-files')

	print("Evaluating on dataset... ")
	results.results_on_dataset = evaluate_on_dataset(eval_files, model_name)
	print("done")
	print("Evaluating per image...")
	results.results_per_image = evaluate_per_image(eval_files, model_name)
	print("done")

	return results

def _compute_raw_results(model_name: str, pred_manager: PredManager) -> RawResults:
	n_images_with_predictions = pred_manager.get_n_images_with_predictions(model_name)
	n_objects_predicted = pred_manager.get_n_objects(model_name)
	class_dist_for_predictions = pred_manager.class_distribution(model_name)

	return RawResults(
		n_images_with_predictions=n_images_with_predictions,
		n_objects_predicted=n_objects_predicted,
		class_dist_for_predictions=class_dist_for_predictions,
	)

def _compute_eval_filters(ann_classes: set[str], pred_classes: set[str]) -> EvalFilters:
	common_classes = [c for c in ann_classes if c in pred_classes]
	pred_classes_ignored = list(pred_classes - common_classes)
	ann_classes_ignored = list(ann_classes - common_classes)

	return EvalFilters(
		classes_considered=common_classes,
		pred_classes_ignored=pred_classes_ignored,
		ann_classes_ignored=ann_classes_ignored,
	)

def _save_results(results: ModelResults, out_dir: Path):
	results_dict = asdict(results)

	class CustomEncoder(json.JSONEncoder):
		def default(self, o: Any) -> Any:
			if type(o) == Annotation or type(o) == Prediction:
				return o.serializable()
			else:
				return super().default(o)

	results_file = out_dir / 'results.json'
	with results_file.open('w') as f:
		json.dump(results_dict, f, indent=4, cls=CustomEncoder)