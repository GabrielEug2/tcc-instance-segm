
import json
from pathlib import Path

from .structures.dataset_info import DatasetInfo
from .structures.results import ImgResults


def group_results_by_img(eval_dir: Path, model_names: list[str]):
	dataset_info_dict = _load_dict(eval_dir / 'dataset-info.json')
	dataset_info = DatasetInfo(**dataset_info_dict)
	img_names = dataset_info.info_per_image.keys()

	n_models = len(model_names)
	
	results_per_img = []
	for img_name in img_names:
		results = {}
		results['img_name'] = img_name

		AP_per_model = {}
		for model in model_names:
			results_for_that_model = _load_dict(eval_dir / model / 'results-per-image' / f'{img_name}.json')
			results_for_that_model = ImgResults(**results_for_that_model)

			AP_per_model[model] = results_for_that_model.AP

		results['AP_per_model'] = AP_per_model
		results['average'] = sum(AP_per_model.values()) / n_models
		results['diff_between_highest_and_lowest'] = max(AP_per_model.values()) - min(AP_per_model.values())

		results_per_img.append(results)

	results_per_img = sorted(results_per_img, key=lambda d: d['diff_between_highest_and_lowest'])
	results_per_img = sorted(results_per_img, key=lambda d: d['average'], reverse=True)
	_save(results_per_img, eval_dir / 'imgs-sorted-by-average-AP.txt')

	results_per_img = sorted(results_per_img, key=lambda d: d['average'], reverse=True)
	results_per_img = sorted(results_per_img, key=lambda d: d['diff_between_highest_and_lowest'])
	_save(results_per_img, eval_dir / 'imgs-sorted-by-diff-between-highest-and-lowest-AP.txt')

def _load_dict(file: Path) -> dict:
	with file.open('r') as f:
		content = json.load(f)
	return content

def _save(results_per_img: list[dict], out_file: Path):
	with out_file.open('w') as f:
		f.write(f'img_name | AP_per_model | average | diff_between_highest_and_lowest')

		for results_dict in results_per_img:
			img_name = results_dict['img_name']
			AP_per_model = results_dict['AP_per_model']
			average = results_dict['average']
			performance_diff = results_dict['diff_between_highest_and_lowest']
			f.write(f'\n{img_name} | {AP_per_model} | {average} | {performance_diff}')