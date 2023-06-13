
import json
from pathlib import Path
import shutil

from segm_lib.core.structures import Annotation, Prediction

from ..structures.dataset_info import DatasetInfo


def group_results_by_img(eval_dir: Path):
	grouped_results_dir = eval_dir / 'results_grouped_by_img'
	grouped_results_dir.mkdir(parents=True, exist_ok=True)

	results_per_img = _sort_by_ap(eval_dir, grouped_results_dir)
	_copy_plots(eval_dir, grouped_results_dir)
	_make_img_summary(results_per_img, grouped_results_dir)

def _sort_by_ap(eval_dir: Path, out_dir: Path):
	dataset_info_dict = _load_dataclass(eval_dir / 'dataset-info.json')
	dataset_info = DatasetInfo(**dataset_info_dict)
	img_names = dataset_info.info_per_image.keys()

	model_names = _get_model_names(eval_dir)
	n_models = len(model_names)
	
	results_per_img = []
	for img_name in img_names:
		results = {}
		results['img_name'] = img_name

		AP_per_model = {}
		for model in model_names:
			results_for_that_model = _load_dataclass(eval_dir / model / 'results-per-image' / f'{img_name}.json')

			AP_per_model[model] = results_for_that_model['AP']

		results['AP_per_model'] = AP_per_model
		results['average'] = sum(AP_per_model.values()) / n_models
		results['diff_between_highest_and_lowest'] = max(AP_per_model.values()) - min(AP_per_model.values())

		results_per_img.append(results)

	results_per_img = sorted(results_per_img, key=lambda d: d['diff_between_highest_and_lowest'], reverse=True)
	results_per_img = sorted(results_per_img, key=lambda d: d['average'], reverse=True)
	_save_results_per_img(results_per_img, out_dir / 'imgs-sorted-by-average-AP.json')

	results_per_img = sorted(results_per_img, key=lambda d: d['average'], reverse=True)
	results_per_img = sorted(results_per_img, key=lambda d: d['diff_between_highest_and_lowest'], reverse=True)
	_save_results_per_img(results_per_img, out_dir / 'imgs-sorted-by-diff-between-highest-and-lowest-AP.json')

	return results_per_img

def _copy_plots(eval_dir: Path, out_dir: Path):
	out_dir.mkdir(parents=True, exist_ok=True)
	model_names = _get_model_names(eval_dir)

	for model in model_names:
		plots = (eval_dir / model / 'results-per-image').glob('*.jpg')
		for plot in plots:
			img_name = plot.stem

			out_dir_for_img = out_dir / img_name
			out_dir_for_img.mkdir(exist_ok=True)
			shutil.copy(plot, out_dir_for_img / f'{model}.jpg')

def _make_img_summary(results_per_img: list[dict], out_dir: Path):
	for results in results_per_img:
		img_name = results['img_name']
		out_file = out_dir / img_name / 'summary.json'

		with out_file.open('w') as f:
			json.dump(results, f, indent=4)

def _load_dataclass(file: Path) -> dict:
	with file.open('r') as f:
		img_results = json.load(f, cls=CustomDecoder)
	return img_results

class CustomDecoder(json.JSONDecoder):
	PREDICTION_KEYS = {'classname', 'mask', 'bbox', 'confidence'}
	ANNOTATION_KEYS = {'classname', 'mask', 'bbox'}

	def __init__(self):
		super().__init__(object_hook=self.obj_to_appropriate_class)

	def obj_to_appropriate_class(self, o):
		if type(o) != dict:
			return o

		unknown_dict = o
		if unknown_dict.keys() == self.PREDICTION_KEYS:
			return Prediction(**unknown_dict)
		elif unknown_dict.keys() == self.ANNOTATION_KEYS:
			return Annotation(**unknown_dict)
		else:
			return unknown_dict
		
RESERVED_NAMES = ['base_files', 'results_grouped_by_img']

def _get_model_names(eval_dir: Path) -> list[str]:
	return [f.name for f in eval_dir.glob('*') if f.is_dir() and f.name not in RESERVED_NAMES]

def _save_results_per_img(results_per_img: list[dict], out_file: Path):
	with out_file.open('w') as f:
		json.dump(results_per_img, f, indent=4)