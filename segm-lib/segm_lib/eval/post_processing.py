
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


def plot_tp_fp_fn(eval_dir, img_dir):
	# something like the detectron plot function...
	# def _plot(self, anns_or_preds: list[Annotation]|list[Prediction], img_file: Path, out_file: Path):
	# 		classnames, scores, masks, boxes = [], [], [], []
	# 		for ann_or_pred in anns_or_preds:
	# 			classnames.append(ann_or_pred.classname)
	# 			scores.append(ann_or_pred.confidence if hasattr(ann_or_pred, 'confidence') else 1.0)

	# 			bin_mask = mask_conversions.rle_to_bin_mask(ann_or_pred.mask)
	# 			masks.append(bin_mask)

	# 			boxes.append(ann_or_pred.bbox)
			
	# 		class_list = list(set(classnames))
	# 		class_ids = []
	# 		for classname in classnames:
	# 			class_ids.append(class_list.index(classname))
	# 		metadata = Metadata()
	# 		metadata.set(thing_classes = class_list)

	# 		boxes_for_api = []
	# 		for box in boxes:
	# 			# pra essa API do Detectron precisa estar em [x1,y1,x2,y2]
	# 			x1, y1, w, h = box
	# 			x2 = x1 + w
	# 			y2 = y1 + h
	# 			boxes_for_api.append([x1, y1, x2, y2])

	# 		img = cv2.imread(str(img_file))
	# 		h, w, _ = img.shape
	# 		instances = Instances((h, w))
	# 		instances.pred_classes = torch.tensor(class_ids, dtype=torch.int)
	# 		instances.scores = torch.tensor(scores, dtype=torch.float)
	# 		instances.pred_masks = torch.stack(masks) if len(masks) >= 1 else torch.tensor([])
	# 		instances.pred_boxes = Boxes(torch.tensor(boxes_for_api, dtype=torch.float))

	# 		v = Visualizer(img, metadata)
	# 		vis_out = v.draw_instance_predictions(instances)
	# 		out_img = vis_out.get_image()

	# 		out_file.parent.mkdir(parents=True, exist_ok=True)
	# 		cv2.imwrite(str(out_file), out_img)

	# but using the tp/fp/fn I saved
	# 	changing the classes to "true_positive", "false_positive" and "false_negative"
	#	or maybe "classname_TP", "classname_FP" amd "classname_FN" (probably this 2nd one)

	# and an unique color for TP/FP/FN
	# 	setting metadata.thing_colors
	# 		thing_colors (list[tuple(r, g, b)]): Pre-defined color (in [0, 255])
	# 		for each thing category. Used for visualization. If not given, random
	# 		colors will be used.
	pass