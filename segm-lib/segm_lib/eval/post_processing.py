
from enum import Enum
import json
from pathlib import Path

from tqdm import tqdm

from segm_lib.core.structures import Prediction, Annotation
from segm_lib.plot.detectron_plot_lib import DetectronPlotLib

from .structures.dataset_info import DatasetInfo
from .structures.results import ImgResults, TP_FP_FN_DetailedInfo


def group_results_by_img(eval_dir: Path):
	dataset_info_dict = _load_img_results(eval_dir / 'dataset-info.json')
	dataset_info = DatasetInfo(**dataset_info_dict)
	img_names = dataset_info.info_per_image.keys()

	model_names = _model_names_from_eval_dir(eval_dir)
	n_models = len(model_names)
	
	results_per_img = []
	for img_name in img_names:
		results = {}
		results['img_name'] = img_name

		AP_per_model = {}
		for model in model_names:
			results_for_that_model = _load_img_results(eval_dir / model / 'results-per-image' / f'{img_name}.json')

			AP_per_model[model] = results_for_that_model.AP

		results['AP_per_model'] = AP_per_model
		results['average'] = sum(AP_per_model.values()) / n_models
		results['diff_between_highest_and_lowest'] = max(AP_per_model.values()) - min(AP_per_model.values())

		results_per_img.append(results)

	results_per_img = sorted(results_per_img, key=lambda d: d['diff_between_highest_and_lowest'])
	results_per_img = sorted(results_per_img, key=lambda d: d['average'], reverse=True)
	_save_results_per_img(results_per_img, eval_dir / 'imgs-sorted-by-average-AP.txt')

	results_per_img = sorted(results_per_img, key=lambda d: d['average'], reverse=True)
	results_per_img = sorted(results_per_img, key=lambda d: d['diff_between_highest_and_lowest'])
	_save_results_per_img(results_per_img, eval_dir / 'imgs-sorted-by-diff-between-highest-and-lowest-AP.txt')


class Colors(Enum):
	GREEN = (11, 181, 85)
	GREY = (140, 140, 140)
	BLUE = (44, 144, 232)
	RED = (230, 25, 25)

def plot_tp_fp_fn(eval_dir: Path, img_dir: Path):
	model_names = _model_names_from_eval_dir(eval_dir)
	detectron_plot_lib = DetectronPlotLib()

	print("Plotting tp/fp/fn...")
	for model in model_names:
		print(f'  \n{model}...')
		model_dir = eval_dir / model
		out_dir = model_dir / 'TP_FP_FN_plot'
		out_dir.mkdir(parents=True, exist_ok=True)

		img_result_files = (model_dir / 'results-per-image').glob('*.json')
		n_imgs = sum(1 for f in (model_dir / 'results-per-image').glob('*.json'))

		for img_result_file in tqdm(img_result_files, total=n_imgs):
			results_for_that_img = _load_img_results(img_result_file)

			objs_to_plot = []
			for classname, tp_list in results_for_that_img.true_positives.list_per_class.items():
				for tp in tp_list:
					classname_for_plot = f'{classname}_TP_det'
					confidence = tp[0].confidence
					mask = tp[0].mask
					bbox = tp[0].bbox
					objs_to_plot.append(Prediction(classname_for_plot, confidence, mask, bbox))

					classname_for_plot = f'{classname}_TP_ann'
					mask = tp[1].mask # black
					bbox = tp[1].bbox
					objs_to_plot.append(Annotation(classname_for_plot, mask, bbox))

			for classname, fp_list in results_for_that_img.false_positives.list_per_class.items():
				classname_for_plot = f'{classname}_FP'
				for fp in fp_list:
					confidence = fp.confidence
					mask = fp.mask
					bbox = fp.bbox
					objs_to_plot.append(Prediction(classname_for_plot, confidence, mask, bbox))

			for classname, fn_list in results_for_that_img.false_negatives.list_per_class.items():
				classname_for_plot = f'{classname}_FN'
				for fn in fn_list:
					mask = fn.mask
					bbox = fn.bbox
					objs_to_plot.append(Annotation(classname_for_plot, mask, bbox))

			class_list = (o.classname for o in objs_to_plot)
			colors_for_plot = {}
			for classname in class_list:
				if classname.endswith('_TP_det'):
					colors_for_plot[classname] = Colors.GREEN.value
				elif classname.endswith('_TP_ann'):
					colors_for_plot[classname] = Colors.GREY.value
				elif classname.endswith('_FP'):
					colors_for_plot[classname] = Colors.BLUE.value
				else: # _FN
					colors_for_plot[classname] = Colors.RED.value

			img_file = next(img_dir.glob(f'*{img_result_file.stem}*'))
			if img_file is None:
				continue
			out_file = out_dir / img_file.name
			detectron_plot_lib.plot(objs_to_plot, img_file, out_file, colors=colors_for_plot)


class CustomDecoder(json.JSONDecoder):
	IMG_RESULT_KEYS = {'anns_considered', 'preds_considered', 'AP', 'true_positives', 'false_positives', 'false_negatives'}
	TP_FP_FN_DETAILEDINFO_KEYS = {'n', 'n_per_class', 'list_per_class'}
	PREDICTION_KEYS = {'classname', 'mask', 'bbox', 'confidence'}
	ANNOTATION_KEYS = {'classname', 'mask', 'bbox'}

	def __init__(self):
		super().__init__(object_hook=self.obj_to_appropriate_class)

	def obj_to_appropriate_class(self, o):
		if type(o) != dict:
			return o

		unknown_dict = o
		if unknown_dict.keys() == self.IMG_RESULT_KEYS:
			return ImgResults(
				anns_considered=None, # doesn't matter, won't use
				preds_considered=None, # doesn't matter, won't use
				AP=unknown_dict['AP'],
				true_positives=self.obj_to_appropriate_class(unknown_dict['true_positives']),
				false_positives=self.obj_to_appropriate_class(unknown_dict['false_positives']),
				false_negatives=self.obj_to_appropriate_class(unknown_dict['false_negatives']),
			)
		elif unknown_dict.keys() == self.TP_FP_FN_DETAILEDINFO_KEYS:
			return TP_FP_FN_DetailedInfo(**unknown_dict)
		elif unknown_dict.keys() == self.PREDICTION_KEYS:
			return Prediction(**unknown_dict)
		elif unknown_dict.keys() == self.ANNOTATION_KEYS:
			return Annotation(**unknown_dict)
		else:
			return unknown_dict

def _load_img_results(file: Path) -> ImgResults:
	with file.open('r') as f:
		img_results = json.load(f, cls=CustomDecoder)
	return img_results

def _save_results_per_img(results_per_img: list[dict], out_file: Path):
	with out_file.open('w') as f:
		f.write(f'img_name | AP_per_model | average | diff_between_highest_and_lowest')

		for results_dict in results_per_img:
			img_name = results_dict['img_name']
			AP_per_model = results_dict['AP_per_model']
			average = results_dict['average']
			performance_diff = results_dict['diff_between_highest_and_lowest']
			f.write(f'\n{img_name} | {AP_per_model} | {average} | {performance_diff}')

def _model_names_from_eval_dir(eval_dir):
	return [f.name for f in eval_dir.glob('*') if f.is_dir() and f.name not in ['base_files', 'plot']]

