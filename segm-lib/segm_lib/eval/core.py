import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..core.managers.ann_manager import AnnManager
from ..core.managers.coco_ann_manager import COCOAnnManager
from ..core.managers.pred_manager import PredManager
from ..core.structures import Annotation, Prediction
from . import cocoapi_wrapper
from .structures.dataset_info import DatasetInfo, ImageInfo
from .structures.eval_files import EvalFiles, EvalFilesForImg
from .structures.model_results import (AnnsOrPredsInfo, DatasetResults,
                                       EvalFilters, ImgResults, ModelResults,
                                       RawResults)


def evaluate_all(pred_dir: Path, ann_dir: Path, coco_ann_file: Path, out_dir: Path):
	pred_manager = PredManager(pred_dir)
	model_names = pred_manager.get_model_names()
	if len(model_names) == 0:
		print(f'No predictions found on "{pred_dir}".')
		return
	print(f'Found predictions for models {model_names}')

	out_dir.mkdir(parents=True, exist_ok=True)

	ann_manager = AnnManager(ann_dir)
	dataset_info = _compute_dataset_info(ann_manager)
	_save_dataset_info(dataset_info, out_dir)

	for model_name in model_names:
		eval_dir = out_dir / model_name
		eval_dir.mkdir(exist_ok=True)
		results = _evaluate_model(model_name, pred_manager, ann_manager, coco_ann_file, eval_dir)
		_save_results(results, eval_dir)

	# TODO parse it to per image

def _compute_dataset_info(ann_manager: AnnManager) -> DatasetInfo:
	print(f'Computing dataset info... ')

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

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def _evaluate_model(
		model_name: str,
		pred_manager: PredManager,
		ann_manager: AnnManager,
		coco_ann_file: Path,
		out_dir: Path
		) -> ModelResults:
	print(f'\nEvaluating model "{model_name}"... ')

	results = ModelResults()
	results.raw_results = _compute_raw_results(model_name, pred_manager)
	results.eval_filters = _compute_eval_filters(ann_manager, pred_manager, model_name)

	eval_files = _prep_eval_files(
		ann_manager,
		pred_manager,
		model_name,
		results.eval_filters,
		coco_ann_file,
		out_dir / 'eval-files'
	)

	print('Evaluating on dataset... ', end='')
	with HiddenPrints():
		results.results_on_dataset = evaluate_on_dataset(eval_files, model_name)
	print('done')
	print('Evaluating per image...')
	with HiddenPrints():
		results.results_per_image = evaluate_per_image(eval_files, model_name)
	print('done')

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

def _compute_eval_filters(ann_manager: AnnManager, pred_manager: PredManager, model_name: str) -> EvalFilters:
	ann_classes = ann_manager.get_classnames()
	pred_classes = pred_manager.get_classnames(model_name)

	common_classes = [c for c in ann_classes if c in pred_classes]
	pred_classes_ignored = list(pred_classes - common_classes)
	ann_classes_ignored = list(ann_classes - common_classes)

	return EvalFilters(
		classes_considered=common_classes,
		pred_classes_ignored=pred_classes_ignored,
		ann_classes_ignored=ann_classes_ignored,
	)

def _prep_eval_files(
		ann_manager: AnnManager,
		pred_manager: PredManager,
		model_name: str,
		eval_filters: EvalFilters,
		original_ann_file: Path,
		out_dir: Path
		) -> EvalFiles:
	classes_to_keep = eval_filters.classes_considered

	custom_anns_dir = out_dir / 'custom_anns'
	ann_manager.filter(custom_anns_dir, classes=classes_to_keep)

	custom_preds_dir = out_dir / 'custom_preds'
	pred_manager.filter(custom_preds_dir, model_name, classes=classes_to_keep)

	# Ideally I would build the ann file from my own annotation format, but the
	# eval API uses some fields which I don't save on mine, like iscrowd. It's
	# easier to just filter the original file too than modifying my format
	# to include all those fields.
	coco_anns_file = out_dir / 'coco_anns.json'
	COCOAnnManager(original_ann_file).filter(coco_anns_file, classes=classes_to_keep)

	coco_preds_file = out_dir / 'coco_preds.json'
	filtered_coco_anns_manager = COCOAnnManager(coco_anns_file)
	classmap = filtered_coco_anns_manager.classmap()
	img_map = filtered_coco_anns_manager.img_map()
	filtered_pred_manager = PredManager(custom_preds_dir)
	filtered_pred_manager.to_coco_format(model_name, img_map, classmap, coco_preds_file)

	eval_files_per_img = {}
	for img_file_name, img_id in img_map.items():
		base_path = out_dir / 'per_image' / img_file_name

		custom_ann_dir_for_img = base_path / 'custom_anns'
		AnnManager(custom_anns_dir).filter(custom_ann_dir_for_img, img_file_name=img_file_name)

		custom_pred_dir_for_img = base_path / 'custom_preds'
		PredManager(custom_preds_dir).filter(custom_pred_dir_for_img, model_name, img_file_name=img_file_name)

		# Tecnicamente eu poderia só usar a API setando E.params.imgIds, o que
		# dispensaria esses arquivos do COCO por imagem, mas eu prefiro "mentir"
		# para API com um dataset de 1 imagem do que depender dessa funcionalidade
		# deles (vai saber o que muda dentro do código)
		coco_ann_file_for_img = base_path / 'coco_anns.json'
		filtered_coco_anns_manager.filter(coco_ann_file_for_img, img_file_name=img_file_name)

		coco_pred_file_for_img = base_path / 'coco_preds.json'
		# classmap is already computed
		img_map = { img_file_name: img_id }
		PredManager(custom_preds_dir).to_coco_format(model_name, img_map, classmap, coco_pred_file_for_img)

		eval_files_per_img[img_file_name] = EvalFilesForImg(
			custom_anns_dir=custom_ann_dir_for_img,
			custom_preds_dir=custom_pred_dir_for_img,
			coco_anns_file=coco_ann_file_for_img,
			coco_preds_file=coco_pred_file_for_img,
		)

	return EvalFiles(
		custom_preds_dir=custom_preds_dir,
		custom_anns_dir=custom_anns_dir,
		coco_preds_file=coco_preds_file,
		coco_anns_file=coco_anns_file,
		per_image=eval_files_per_img
	)

def evaluate_on_dataset(eval_files: EvalFiles, model_name: str) -> DatasetResults:
	dataset_results = DatasetResults()

	filtered_anns_manager = AnnManager(eval_files.custom_anns_dir)
	n_anns_considered = filtered_anns_manager.get_n_objects()
	class_dist_anns_considered = filtered_anns_manager.class_distribution()
	dataset_results.anns_considered = AnnsOrPredsInfo(
		n=n_anns_considered,
		class_dist=class_dist_anns_considered,
	)

	filtered_preds_manager = PredManager(eval_files.custom_preds_dir)
	n_preds_considered = filtered_preds_manager.get_n_objects(model_name)
	class_dist_preds_considered = filtered_preds_manager.class_distribution(model_name)
	dataset_results.preds_considered = AnnsOrPredsInfo(
		n=n_preds_considered,
		class_dist=class_dist_preds_considered,
	)

	results_from_coco_api = cocoapi_wrapper.eval(
		eval_files.coco_anns_file,
		eval_files.coco_preds_file,
		detailed=False
	)
	dataset_results.mAP = results_from_coco_api.mAP
	dataset_results.true_positives = results_from_coco_api.true_positives
	dataset_results.false_positives = results_from_coco_api.false_positives
	dataset_results.false_negatives = results_from_coco_api.false_negatives

	return dataset_results

def evaluate_per_image(eval_files: EvalFiles, model_name: str) -> dict[str, ImgResults]:
	results_per_img = {}

	coco_ann_manager = COCOAnnManager(eval_files.coco_anns_file)
	img_map = coco_ann_manager.img_map()

	for img_name, img_id in tqdm(img_map.items()):
		img_results = ImgResults()

		filtered_anns_manager = AnnManager(eval_files.per_image[img_name].custom_anns_dir)
		n_anns_considered = filtered_anns_manager.get_n_objects()
		class_dist_anns_considered = filtered_anns_manager.class_distribution()
		img_results.anns_considered = AnnsOrPredsInfo(
			n=n_anns_considered,
			class_dist=class_dist_anns_considered,
		)
				
		filtered_preds_manager = PredManager(eval_files.per_image[img_name].custom_preds_dir)
		n_preds_considered = filtered_preds_manager.get_n_objects(model_name)
		class_dist_preds_considered = filtered_preds_manager.class_distribution(model_name)
		img_results.preds_considered = AnnsOrPredsInfo(
			n=n_preds_considered,
			class_dist=class_dist_preds_considered,
		)
	
		results_from_coco_api = cocoapi_wrapper.eval(
			eval_files.per_image[img_name].coco_anns_file,
			eval_files.per_image[img_name].coco_preds_file,
			detailed=True
		)
		img_results.mAP = results_from_coco_api.mAP
		img_results.true_positives = results_from_coco_api.true_positives
		img_results.false_positives = results_from_coco_api.false_positives
		img_results.false_negatives = results_from_coco_api.false_negatives
	
		results_per_img[img_name] = img_results

	return results_per_img

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