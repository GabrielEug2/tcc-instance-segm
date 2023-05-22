
import json
from pathlib import Path

from .structures.dataset_info import DatasetInfo
from .structures.results import ImgResults


def group_results_by_img(eval_dir: Path, model_names: list[str]):
	dataset_info_dict = _load(eval_dir / 'dataset-info.json')
	dataset_info = DatasetInfo(**dataset_info_dict)
	n_models = len(model_names)
	
	results_per_img = {}
	for img_name in dataset_info.info_per_image:
		results_for_that_img = {
			'raw': {},
			'stats': {},
		}
		
		for model in model_names:
			results_for_that_model = _load(eval_dir / model / 'results-per-image' / f'{img_name}.json')
			results_for_that_model = ImgResults(**results_for_that_model)

			results_for_that_img['raw'][model] = results_for_that_model.mAP

		results_for_that_img['stats']['average'] = sum(results_for_that_img['raw'].values()) / n_models

		min_ap = min(results_for_that_img['raw'].values())
		max_ap = max(results_for_that_img['raw'].values())
		results_for_that_img['stats']['difference_between_highest_and_lowest_model'] = max_ap - min_ap

		results_per_img[img_name] = results_for_that_img

	info_to_write = {
		'sorted_by_average (descending)': dict(sorted(
			results_per_img.items(),
			key=lambda d: d[1]['stats']['average'],
			reverse=True
		)),
		'sorted_by_higest_ap_average (descending)': dict(sorted(
			results_per_img.items(),
			key=lambda d: d[1]['stats']['difference_between_highest_and_lowest_model'],
			reverse=True
		)),
	}

	out_file = eval_dir / 'mAP-by-image.json'
	with out_file.open('w') as f:
		json.dump(info_to_write, f, indent=4)

def _load(file: Path) -> dict:
	with file.open('r') as f:
		content = json.load(f)
	return content