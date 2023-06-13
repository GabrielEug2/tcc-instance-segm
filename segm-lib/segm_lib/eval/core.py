import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from segm_lib.core.classname_normalization import normalize_classname
from segm_lib.core.managers import (AnnManager, COCOAnnManager,
                                    MultiModelPredManager,
                                    SingleModelPredManager)
from segm_lib.core.structures import Annotation, Prediction
from segm_lib.eval.structures.eval_filters import EvalFilters
from tqdm import tqdm

from . import cocoapi_wrapper, post_processing
from .structures.dataset_info import DatasetInfo, ImageInfo
from .structures.eval_files import COCOFiles, EvalFiles
from .structures.eval_results import EvalResults
from .structures.raw_results import RawResults


def evaluate_all(
		pred_dir: Path,
		ann_dir: Path,
		possible_classes_dir: Path,
		coco_ann_file: Path,
		img_dir: Path,
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
	pred_manager = MultiModelPredManager(eval_pred_dir)
	pred_manager.normalize_classnames()
	ann_manager = AnnManager(eval_ann_dir)
	ann_manager.normalize_classnames()
	# the COCO file is only normalized where it's gonna be used,
	# to avoid having to load it twice or keep it on memory while
	# it's not needed
	print('done')

	print(f'Computing dataset info... ', end='', flush=True)
	dataset_info = _compute_dataset_info(ann_manager)
	_save_dataclass(dataset_info, out_dir / 'dataset-info.json')
	del dataset_info
	print('done')

	model_names = pred_manager.get_model_names()
	if len(model_names) == 0:
		print(f'No predictions found on "{pred_dir}".')
		return
	print(f'Found predictions for models {model_names}')

	for model_name in model_names:
		print(f'\nEvaluating {model_name}... ')
		eval_out_dir = out_dir / model_name
		eval_out_dir.mkdir(exist_ok=True)

		model_pred_manager = pred_manager.get_manager(model_name)
		with (possible_classes_dir / f'{model_name}.json').open('r') as f:
			possible_pred_classes = json.load(f)
		possible_pred_classes = {normalize_classname(c) for c in set(possible_pred_classes)}
		_evaluate_model(
			model_pred_manager,
			possible_pred_classes,
			ann_manager,
			eval_coco_ann_file,
			eval_out_dir,
			img_dir
			)

	print('\nPost-processing... ')
	print('  Grouping results by img... ')
	post_processing.group_results_by_img(out_dir)
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
			class_dist=_sort_by_key(class_dist_on_img)
		)

	return DatasetInfo(
		n_images=n_images,
		n_objects=n_objects,
		class_dist=_sort_by_key(class_dist),
		info_per_image=info_per_img
	)

def _evaluate_model(
		model_pred_manager: SingleModelPredManager,
		possible_pred_classes: set[str],
		ann_manager: AnnManager,
		coco_ann_file: Path,
		model_dir: Path,
		img_dir: Path
		):
	print(f'Computing raw results... ', end='', flush=True)
	raw_results = _compute_raw_results(model_pred_manager)
	_save_dataclass(raw_results, model_dir / 'raw-results.json')
	del raw_results
	print('done')

	print(f'Computing eval filters... ', end='', flush=True)
	eval_filters = _compute_eval_filters(ann_manager, model_pred_manager, possible_pred_classes)
	_save_dataclass(eval_filters, model_dir / 'eval-filters.json')
	print('done')

	print(f'Preparing files for evaluation... ', end='', flush=True)
	eval_files = _prep_eval_files(
		ann_manager,
		model_pred_manager,
		eval_filters.evaluatable_classes,
		coco_ann_file,
		model_dir / 'eval_files'
	)
	print('done')
	del eval_filters

	print('  Evaluating on dataset... ', end='', flush=True)
	_evaluate_on_dataset(eval_files, model_dir / 'results-on-dataset.json')
	print('done')

	print('  Evaluating per image...', flush=True)
	_evaluate_per_image(eval_files, model_dir / 'results-per-image', img_dir)
	print('done')

def _compute_raw_results(model_pred_manager: SingleModelPredManager) -> RawResults:
	n_images_with_predictions = model_pred_manager.get_n_images_with_predictions()
	n_objects_predicted = model_pred_manager.get_n_objects()
	class_dist_for_predictions = model_pred_manager.class_distribution()

	return RawResults(
		n_images_with_preds=n_images_with_predictions,
		n_objects_predicted=n_objects_predicted,
		class_dist_on_preds=_sort_by_key(class_dist_for_predictions),
	)

def _compute_eval_filters(
		ann_manager: AnnManager,
		model_pred_manager: SingleModelPredManager,
		possible_pred_classes: list[str]
		) -> EvalFilters:
	ann_classes = ann_manager.get_classnames()
	pred_classes = model_pred_manager.get_classnames()

	evaluatable_classes = {c for c in ann_classes if c in possible_pred_classes}
	predicted_classes_impossible_to_evaluate = pred_classes - evaluatable_classes
	annotated_classes_impossible_to_predict = ann_classes - evaluatable_classes
	predictable_classes_impossible_to_evaluate = possible_pred_classes - evaluatable_classes

	return EvalFilters(
		evaluatable_classes=evaluatable_classes,
		predicted_classes_impossible_to_evaluate=predicted_classes_impossible_to_evaluate,
		annotated_classes_impossible_to_predict=annotated_classes_impossible_to_predict,
		predictable_classes_impossible_to_evaluate=predictable_classes_impossible_to_evaluate,
	)

def _prep_eval_files(
		ann_manager: AnnManager,
		pred_manager: SingleModelPredManager,
		classes_to_keep: set[str],
		original_ann_file: Path,
		out_dir: Path
		) -> EvalFiles:
	filtered_anns_dir = out_dir / 'filtered_anns'
	ann_manager.filter(filtered_anns_dir, classes=classes_to_keep)
	filtered_ann_manager = AnnManager(filtered_anns_dir)

	filtered_preds_dir = out_dir / 'filtered_preds'
	pred_manager.filter(filtered_preds_dir, classes=classes_to_keep)
	filtered_pred_manager = SingleModelPredManager(filtered_preds_dir)

	coco_files = {}

	# Ideally I would build the ann file from my own annotation format, but the
	# eval API uses some fields which I don't save on mine, like iscrowd. It's
	# # easier to just filter the original file too than modifying my format
	# # to include all those fields.
	filtered_coco_anns_file = out_dir / 'filtered_coco_anns.json'
	coco_ann_manager = COCOAnnManager(original_ann_file)
	coco_ann_manager.normalize_classnames()
	coco_ann_manager.filter(filtered_coco_anns_file, classes=classes_to_keep)
	del coco_ann_manager

	filtered_coco_ann_manager = COCOAnnManager(filtered_coco_anns_file)
	classmap = filtered_coco_ann_manager.classmap()
	img_map = filtered_coco_ann_manager.img_map()

	filtered_coco_preds_file = out_dir / 'filtered_coco_preds.json'
	filtered_pred_manager = SingleModelPredManager(filtered_preds_dir)
	filtered_pred_manager.to_coco_format(img_map, classmap, filtered_coco_preds_file)

	coco_files['dataset'] = COCOFiles(
		anns_file=filtered_coco_anns_file,
		preds_file=filtered_coco_preds_file
	)

	for img_name in img_map.keys():
		img_eval_dir = out_dir / 'per_image' / img_name

		# Tecnicamente eu poderia só usar a API setando E.params.imgIds, o que
		# dispensaria esses arquivos do COCO por imagem, mas eu prefiro "mentir"
		# para API com um dataset de 1 imagem do que depender dessa funcionalidade
		# deles (vai saber o que muda dentro do código)
		coco_ann_file_for_img = img_eval_dir / 'coco_anns.json'
		filtered_coco_ann_manager.filter(coco_ann_file_for_img, img_name=img_name)

		coco_pred_file_for_img = img_eval_dir / 'coco_preds.json'
		tmp_dir = Path('tmp')
		filtered_pred_manager.filter(tmp_dir, img_name=img_name)
		SingleModelPredManager(tmp_dir).to_coco_format(img_map, classmap, coco_pred_file_for_img)
		shutil.rmtree(str(tmp_dir))

		coco_files[img_name] = COCOFiles(
			anns_file=coco_ann_file_for_img,
			preds_file=coco_pred_file_for_img,
		)

	return EvalFiles(
		anns_dir=filtered_ann_manager.root_dir,
		preds_dir=filtered_pred_manager.model_dir,
		coco_files=coco_files
	)

def _evaluate_on_dataset(eval_files: EvalFiles, out_file: Path):
	filtered_anns_manager = AnnManager(eval_files.anns_dir)
	n_anns_considered = filtered_anns_manager.get_n_objects()
	class_dist_anns_considered = filtered_anns_manager.class_distribution()

	filtered_preds_manager = SingleModelPredManager(eval_files.preds_dir)
	n_preds_considered = filtered_preds_manager.get_n_objects()
	class_dist_preds_considered = filtered_preds_manager.class_distribution()

	with HiddenPrints():
		api_results = cocoapi_wrapper.eval(
			eval_files.coco_files['dataset'].anns_file,
			eval_files.coco_files['dataset'].preds_file,
			detailed=False
		)

	results = EvalResults(
		n_anns_considered=n_anns_considered,
		class_dist_anns_considered=_sort_by_key(class_dist_anns_considered),
		n_preds_considered=n_preds_considered,
		class_dist_preds_considered=_sort_by_key(class_dist_preds_considered),

		AP=api_results.AP,
		true_positives=api_results.true_positives,
		false_positives=api_results.false_positives,
		false_negatives=api_results.false_negatives,
	)
	_save_dataclass(results, out_file)

def _evaluate_per_image(eval_files: EvalFiles, out_dir: Path, img_dir: Path):
	filtered_anns_manager = AnnManager(eval_files.anns_dir)
	filtered_preds_manager = SingleModelPredManager(eval_files.preds_dir)

	img_names = (k for k in eval_files.coco_files if k != 'dataset')
	n_imgs = sum((1 for k in eval_files.coco_files), start=-1) # -1 por causa do 'dataset'

	for img_name in tqdm(img_names, total=n_imgs):
		n_anns_considered = filtered_anns_manager.get_n_objects(img_name=img_name)
		class_dist_anns_considered = filtered_anns_manager.class_distribution(img_name=img_name)

		n_preds_considered = filtered_preds_manager.get_n_objects(img_name=img_name)
		class_dist_preds_considered = filtered_preds_manager.class_distribution(img_name=img_name)

		with HiddenPrints():
			api_results = cocoapi_wrapper.eval(
				eval_files.coco_files[img_name].anns_file,
				eval_files.coco_files[img_name].preds_file,
				detailed=True
			)

		results_for_img = EvalResults(
			n_anns_considered=n_anns_considered,
			class_dist_anns_considered=_sort_by_key(class_dist_anns_considered),
			n_preds_considered=n_preds_considered,
			class_dist_preds_considered=_sort_by_key(class_dist_preds_considered),

			AP=api_results.AP,
			true_positives=api_results.true_positives,
			false_positives=api_results.false_positives,
			false_negatives=api_results.false_negatives,
		)
		eval_out_file = out_dir / f'{img_name}.json'
		_save_dataclass(results_for_img, eval_out_file)

		plot_out_file = out_dir / f'{img_name}.jpg'
		img_file = img_dir / f'{img_name}.jpg'
		post_processing.plot_tps_fps_fns(results_for_img, plot_out_file, img_file)

def _sort_by_key(my_dict):
	return dict(sorted(my_dict.items(), key=lambda i: i[0]))

class HiddenPrints:
	# Solution from here:
	# https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
	def __enter__(self):
		self._original_stdout = sys.stdout
		sys.stdout = open(os.devnull, 'w')

	def __exit__(self, exc_type, exc_val, exc_tb):
		sys.stdout.close()
		sys.stdout = self._original_stdout

class CustomEncoder(json.JSONEncoder):
	def default(self, o: Any) -> Any:
		if type(o) == Annotation or type(o) == Prediction:
			return o.serializable()
		if type(o) == set:
			return list(o)
		else:
			return super().default(o)

def _save_dataclass(dataclass_obj: DatasetInfo|RawResults|EvalFilters|EvalResults, out_file: Path):
	out_file.parent.mkdir(parents=True, exist_ok=True)
	with out_file.open('w') as f:
		json.dump(asdict(dataclass_obj), f, indent=4, cls=CustomEncoder)
