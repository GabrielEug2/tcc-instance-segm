import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..core.managers.ann_manager import AnnManager
from ..core.managers.coco_ann_manager import COCOAnnManager
from ..core.managers.pred_manager import PredManager
from ..core.structures import Annotation, Prediction
from ..core.classname_normalization import normalize_classname
from . import cocoapi_wrapper
from . import post_processing
from .structures.dataset_info import DatasetInfo, ImageInfo
from .structures.eval_files import EvalFiles, EvalFilesForImg
from .structures.results import (RawResults, EvalFilters, DatasetResults,
				                 AnnsOrPredsInfo, ImgResults)


def evaluate_all(
		pred_dir: Path,
		ann_dir: Path,
		possible_classes_dir: Path,
		coco_ann_file: Path,
		out_dir: Path
		):
	out_dir.mkdir(parents=True, exist_ok=True)

	# A copy because I'll need to modify a few things (like normalize
	# the classnames) and I don't want to modify the original files
	print("Copying necessary files... ", end='', flush=True)
	eval_files_base_dir = out_dir / 'base_files'
	eval_pred_dir = eval_files_base_dir / 'preds'
	eval_ann_dir = eval_files_base_dir / 'anns'
	eval_coco_ann_file = eval_files_base_dir / 'coco_anns.json'
	shutil.copytree(pred_dir, eval_pred_dir, dirs_exist_ok=True)
	shutil.copytree(ann_dir, eval_ann_dir, dirs_exist_ok=True)
	shutil.copy(coco_ann_file, eval_coco_ann_file)
	print('done')

	print("Normalizing class names... ", end='', flush=True)
	pred_manager = PredManager(eval_pred_dir)
	pred_manager.normalize_classnames()
	ann_manager = AnnManager(eval_ann_dir)
	ann_manager.normalize_classnames()
	# eval_coco_ann_file is only normalized where it's gonna be used,
	# to avoid having to load it twice or keep it on memory while it's
	# not needed
	print('done')

	print(f'Computing dataset info... ', end='', flush=True)
	dataset_info = _compute_dataset_info(ann_manager)
	_save_dict(asdict(dataset_info), out_dir / 'dataset-info.json')
	del dataset_info
	print('done')

	model_names = pred_manager.get_model_names()
	if len(model_names) == 0:
		print(f'No predictions found on "{pred_dir}".')
		return
	print(f'Found predictions for models {model_names}')

	for model_name in model_names:
		_evaluate_model(
			model_name,
			pred_manager,
			possible_classes_dir,
			ann_manager,
			eval_coco_ann_file,
			out_dir
			)

	print(f'Post-processing... ', end='', flush=True)
	post_processing.group_results_by_img(out_dir, model_names)
	print('done')

def _compute_dataset_info(ann_manager: AnnManager) -> DatasetInfo:
	n_images = ann_manager.get_n_images()
	n_objects = ann_manager.get_n_objects()
	class_dist = ann_manager.class_distribution()

	info_per_img = {}
	for img_name in ann_manager.get_img_names():
		class_dist_on_img = ann_manager.class_distribution(img_name)
		n_objects_on_img = sum(class_dist_on_img.values())

		info_per_img[img_name] = ImageInfo(
			n_objects=n_objects_on_img,
			class_dist=_dict_sorted_by_key(class_dist_on_img)
		)

	return DatasetInfo(
		n_images=n_images,
		n_objects=n_objects,
		class_dist=_dict_sorted_by_key(class_dist),
		info_per_image=info_per_img
	)

class HiddenPrints:
	# Solution from here:
	# https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
	def __enter__(self):
		self._original_stdout = sys.stdout
		sys.stdout = open(os.devnull, 'w')

	def __exit__(self, exc_type, exc_val, exc_tb):
		sys.stdout.close()
		sys.stdout = self._original_stdout

def _evaluate_model(
		model_name: str,
		pred_manager: PredManager,
		possible_classes_dir: Path,
		ann_manager: AnnManager,
		coco_ann_file: Path,
		out_dir: Path
		):
	print(f'\nEvaluating {model_name}... ')
	model_dir = out_dir / model_name
	model_dir.mkdir(exist_ok=True)
	
	print(f'Computing raw results... ', end='', flush=True)
	raw_results = _compute_raw_results(model_name, pred_manager)
	_save_dict(asdict(raw_results), model_dir / 'raw-results.json')
	print('done')

	print(f'Computing eval filters... ', end='', flush=True)
	eval_filters = _compute_eval_filters(ann_manager, possible_classes_dir, model_name)
	_save_dict(asdict(eval_filters), model_dir / 'eval-filters.json')
	print('done')

	print(f'Preparing files for evaluation... ', end='', flush=True)
	eval_files = _prep_eval_files(
		ann_manager,
		pred_manager,
		model_name,
		eval_filters,
		coco_ann_file,
		model_dir / 'eval_files'
	)
	print('done')

	print('  Evaluating on dataset... ', end='', flush=True)
	with HiddenPrints():
		results_on_dataset = _evaluate_on_dataset(eval_files, model_name)
	_save_dict(asdict(results_on_dataset), model_dir / 'results-on-dataset.json')
	print('done')

	print('  Evaluating per image...', flush=True)
	with HiddenPrints():
		_evaluate_per_image(eval_files, model_name, model_dir / 'results-per-image')
	print('done')

def _compute_raw_results(model_name: str, pred_manager: PredManager) -> RawResults:
	n_images_with_predictions = pred_manager.get_n_images_with_predictions(model_name)
	n_objects_predicted = pred_manager.get_n_objects(model_name)
	class_dist_for_predictions = pred_manager.class_distribution(model_name)

	return RawResults(
		n_images_with_predictions=n_images_with_predictions,
		n_objects_predicted=n_objects_predicted,
		class_dist_for_predictions=_dict_sorted_by_key(class_dist_for_predictions),
	)

def _compute_eval_filters(ann_manager: AnnManager, possible_classes_dir: Path, model_name: str) -> EvalFilters:
	ann_classes = ann_manager.get_classnames()
	with (possible_classes_dir / f'{model_name}.json').open('r') as f:
		possible_pred_classes = json.load(f)
	possible_pred_classes = {normalize_classname(c) for c in set(possible_pred_classes)}

	evaluatable_classes = {c for c in ann_classes if c in possible_pred_classes}
	pred_classes_ignored = list(possible_pred_classes - evaluatable_classes)
	ann_classes_ignored = list(ann_classes - evaluatable_classes)

	return EvalFilters(
		classes_considered=list(evaluatable_classes),
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

	filtered_anns_dir = out_dir / 'filtered_anns'
	ann_manager.filter(filtered_anns_dir, classes=classes_to_keep)

	filtered_preds_dir = out_dir / 'filtered_preds'
	pred_manager.filter(filtered_preds_dir, model_name, classes=classes_to_keep)

	# Ideally I would build the ann file from my own annotation format, but the
	# eval API uses some fields which I don't save on mine, like iscrowd. It's
	# easier to just filter the original file too than modifying my format
	# to include all those fields.
	filtered_coco_anns_file = out_dir / 'filtered_coco_anns.json'
	coco_ann_manager = COCOAnnManager(original_ann_file)
	coco_ann_manager.normalize_classnames()
	coco_ann_manager.filter(filtered_coco_anns_file, classes=classes_to_keep)
	del coco_ann_manager

	filtered_coco_preds_file = out_dir / 'filtered_coco_preds.json'
	filtered_coco_anns_manager = COCOAnnManager(filtered_coco_anns_file)
	classmap = filtered_coco_anns_manager.classmap()
	img_map = filtered_coco_anns_manager.img_map()
	filtered_pred_manager = PredManager(filtered_preds_dir)
	filtered_pred_manager.to_coco_format(model_name, img_map, classmap, filtered_coco_preds_file)

	eval_files_per_img = {}
	for img_name, img_id in img_map.items():
		img_eval_dir = out_dir / 'per_image' / img_name

		# Tecnicamente eu poderia só usar a API setando E.params.imgIds, o que
		# dispensaria esses arquivos do COCO por imagem, mas eu prefiro "mentir"
		# para API com um dataset de 1 imagem do que depender dessa funcionalidade
		# deles (vai saber o que muda dentro do código)
		coco_ann_file_for_img = img_eval_dir / 'coco_anns.json'
		filtered_coco_anns_manager.filter(coco_ann_file_for_img, img_name=img_name)

		coco_pred_file_for_img = img_eval_dir / 'coco_preds.json'
		# classmap is already computed
		img_map = { img_name: img_id }
		PredManager(filtered_preds_dir).to_coco_format(model_name, img_map, classmap, coco_pred_file_for_img)

		eval_files_per_img[img_name] = EvalFilesForImg(
			filtered_coco_anns_file=coco_ann_file_for_img,
			filtered_coco_preds_file=coco_pred_file_for_img,
		)

	return EvalFiles(
		filtered_preds_dir=filtered_preds_dir,
		filtered_anns_dir=filtered_anns_dir,
		filtered_coco_preds_file=filtered_coco_preds_file,
		filtered_coco_anns_file=filtered_coco_anns_file,
		per_image=eval_files_per_img
	)

def _evaluate_on_dataset(eval_files: EvalFiles, model_name: str) -> DatasetResults:
	dataset_results = DatasetResults()

	filtered_anns_manager = AnnManager(eval_files.filtered_anns_dir)
	n_anns_considered = filtered_anns_manager.get_n_objects()
	class_dist_anns_considered = filtered_anns_manager.class_distribution()
	dataset_results.anns_considered = AnnsOrPredsInfo(
		n=n_anns_considered,
		class_dist=_dict_sorted_by_key(class_dist_anns_considered),
	)

	filtered_preds_manager = PredManager(eval_files.filtered_preds_dir)
	n_preds_considered = filtered_preds_manager.get_n_objects(model_name)
	class_dist_preds_considered = filtered_preds_manager.class_distribution(model_name)
	dataset_results.preds_considered = AnnsOrPredsInfo(
		n=n_preds_considered,
		class_dist=_dict_sorted_by_key(class_dist_preds_considered),
	)

	results_from_coco_api = cocoapi_wrapper.eval(
		eval_files.filtered_coco_anns_file,
		eval_files.filtered_coco_preds_file,
		detailed=False
	)
	dataset_results.AP = results_from_coco_api.AP
	dataset_results.true_positives = results_from_coco_api.true_positives
	dataset_results.false_positives = results_from_coco_api.false_positives
	dataset_results.false_negatives = results_from_coco_api.false_negatives

	return dataset_results

def _evaluate_per_image(eval_files: EvalFiles, model_name: str, out_dir: Path):
	filtered_anns_manager = AnnManager(eval_files.filtered_anns_dir)
	filtered_preds_manager = PredManager(eval_files.filtered_preds_dir)

	img_names = eval_files.per_image.keys()
	for img_name in tqdm(img_names):
		img_results = ImgResults()

		n_anns_considered = filtered_anns_manager.get_n_objects(img_name=img_name)
		class_dist_anns_considered = filtered_anns_manager.class_distribution(img_name=img_name)
		img_results.anns_considered = AnnsOrPredsInfo(
			n=n_anns_considered,
			class_dist=_dict_sorted_by_key(class_dist_anns_considered),
		)
				
		n_preds_considered = filtered_preds_manager.get_n_objects(model_name, img_name=img_name)
		class_dist_preds_considered = filtered_preds_manager.class_distribution(model_name, img_name=img_name)
		img_results.preds_considered = AnnsOrPredsInfo(
			n=n_preds_considered,
			class_dist=_dict_sorted_by_key(class_dist_preds_considered),
		)
	
		results_from_coco_api = cocoapi_wrapper.eval(
			eval_files.per_image[img_name].filtered_coco_anns_file,
			eval_files.per_image[img_name].filtered_coco_preds_file,
			detailed=True
		)
		img_results.AP = results_from_coco_api.AP
		img_results.true_positives = results_from_coco_api.true_positives
		img_results.false_positives = results_from_coco_api.false_positives
		img_results.false_negatives = results_from_coco_api.false_negatives
	
		_save_dict(asdict(img_results), out_dir / f'{img_name}.json')

def _save_dict(result_dict: dict, out_file: Path):
	class CustomEncoder(json.JSONEncoder):
		def default(self, o: Any) -> Any:
			if type(o) == Annotation or type(o) == Prediction:
				return o.serializable()
			else:
				return super().default(o)
			
	out_file.parent.mkdir(parents=True, exist_ok=True)
	with out_file.open('w') as f:
		json.dump(result_dict, f, indent=4, cls=CustomEncoder)

def _dict_sorted_by_key(my_dict):
	return dict(sorted(my_dict.items(), key=lambda i: i[0]))