from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import shutil
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from segm_lib.annotations import Annotations
from segm_lib.predictions import Predictions
from segm_lib.coco_annotations import COCOAnnotations

@dataclass
class ImageInfo:
	n_objects: int = 0
	class_dist: dict[str, int] = field(default_factory=dict)

@dataclass
class DatasetInfo:
	n_images: int = 0
	n_objects: int = 0
	class_dist: dict[str, int] = field(default_factory=dict)

	info_per_image: dict[str, ImageInfo] = field(default_factory=dict)

@dataclass
class RawResults:
	n_images_with_predictions: int = 0
	n_objects_predicted: int = 0
	class_dist: dict[str, int] = field(default_factory=dict)

@dataclass
class EvalFilters:
	classes_considered: list[str] = field(default_factory=list)
	pred_classes_ignored: list[str] = field(default_factory=list)
	ann_classes_ignored: list[str] = field(default_factory=list)

@dataclass
class DatasetResults:
	n_anns_considered: int = 0
	n_preds_considered: int = 0

	mAP: float = 0.0
	n_true_positives: int = 0
	n_false_positives: int = 0
	n_false_negatives: int = 0

@dataclass
class ImageResults:
	n_anns_considered: int = 0
	n_preds_considered: int = 0

	mAP: float = 0.0
	true_positives: list[dict] = field(default_factory=list)
	false_positives: list[dict] = field(default_factory=list)
	false_negatives: list[dict] = field(default_factory=list)
	
@dataclass
class ModelResults:
	raw_results: RawResults = None
	eval_filters: EvalFilters = None
	results_on_dataset: DatasetResults = None
	results_per_image: dict[str, ImageResults] = field(default_factory=dict)

@dataclass
class EvalFiles:
	custom_preds_dir: Path = None
	custom_anns_dir: Path = None
	coco_preds_file: Path = None
	coco_anns_file: Path = None
	coco_pred_file_per_image: dict[str, Path] = None
	coco_ann_file_per_image: dict[str, Path] = None
	
def evaluate_all(pred_dir: Path, ann_dir: Path, ann_file: Path, out_dir: Path):
	out_dir.mkdir(parents=True, exist_ok=True)

	ann_manager = Annotations(ann_dir)
	dataset_info = _compute_dataset_info(ann_manager)
	_save_dataset_info(dataset_info, out_dir)

	pred_manager = Predictions(pred_dir)
	model_names = pred_manager.get_model_names()
	for model_name in model_names:
		eval_dir = out_dir / model_name
		eval_dir.mkdir(exist_ok=True)
		results = _evaluate_model(model_name, pred_manager, ann_manager, ann_file, eval_dir)
		_save_results(results, eval_dir)

	# Save it per model, then parse it to per image

def _compute_dataset_info(ann_manager: Annotations) -> DatasetInfo:
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
	with dataset_info_file.open('w') as f:
		json.dump(dataset_info_dict, f)

def _evaluate_model(
	model_name: str,
	pred_manager: Predictions,
	ann_manager: Annotations,
	ann_file: Path,
	out_dir: Path
) -> ModelResults:
	results = ModelResults()
	results.raw_results = _compute_raw_results(model_name, pred_manager)

	ann_classes = ann_manager.get_classnames()
	pred_classes = pred_manager.get_classnames(model_name)
	results.eval_filters = _compute_eval_filters(ann_classes, pred_classes)

	common_classes = results.eval_filters.classes_considered
	eval_files = _prep_files_for_eval(model_name, pred_manager, ann_manager, ann_file, common_classes, out_dir)

	results.results_on_dataset = _evaluate_on_dataset(eval_files, model_name)
	results.results_per_image = _evaluate_per_image(eval_files, model_name)

	# for img, img_results in results.results_per_image.items():
	# 	for k, v in img_results.__dataclass_fields__.items():
	# 		print(k, getattr(img_results, v.name), v.type)

	return results

def _compute_raw_results(model_name: str, pred_manager: Predictions) -> RawResults:
	n_images_with_predictions = pred_manager.get_n_images_with_predictions(model_name)
	n_objects_predicted = pred_manager.get_n_objects(model_name)
	pred_class_dist = pred_manager.class_distribution(model_name)

	return RawResults(
		n_images_with_predictions=n_images_with_predictions,
		n_objects_predicted=n_objects_predicted,
		class_dist=pred_class_dist,
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

def _prep_files_for_eval(
	model_name: str,
	pred_manager: Predictions,
	ann_manager: Annotations,
	original_ann_file: Path,
	common_classes: list[str],
	out_dir: Path
):
	actual_out_dir = out_dir / 'eval-files'
	
	custom_preds_dir = actual_out_dir / 'custom_preds'
	pred_manager.filter(custom_preds_dir, model_name, classes=common_classes)

	custom_anns_dir = actual_out_dir / 'custom_anns'
	ann_manager.filter(custom_anns_dir, classes=common_classes)

	# Ideally I would build the ann file from my own annotation format, but the
	# eval API uses some fields which I don't save on mine, like iscrowd. It's
	# easier to just filter the original file too than modifying my format
	# to include all those fields.
	coco_anns_file = actual_out_dir / 'coco_anns.json'
	COCOAnnotations(original_ann_file).filter(coco_anns_file, classes=common_classes)

	coco_preds_file = actual_out_dir / 'coco_preds.json'
	filtered_coco_anns_manager = COCOAnnotations(coco_anns_file)
	classmap = filtered_coco_anns_manager.classmap()
	img_map = filtered_coco_anns_manager.img_map()
	filtered_pred_manager = Predictions(custom_preds_dir)
	filtered_pred_manager.to_coco_format(model_name, img_map, classmap, coco_preds_file)

	coco_ann_file_per_image = {}
	coco_pred_file_per_image = {}
	for img_file_name, img_id in img_map.items():
		ann_file_for_img = actual_out_dir / 'per_image' / img_file_name / "coco_anns.json"
		COCOAnnotations(coco_anns_file).filter(ann_file_for_img, img_file_name=img_file_name)

		pred_file_for_img = actual_out_dir / 'per_image' / img_file_name / "coco_preds.json"
		filtered_coco_anns_manager = COCOAnnotations(ann_file_for_img)
		classmap = filtered_coco_anns_manager.classmap()
		img_map = { img_file_name: img_id }
		Predictions(custom_preds_dir).to_coco_format(model_name, img_map, classmap, pred_file_for_img)

		coco_ann_file_per_image[img_file_name] = ann_file_for_img
		coco_pred_file_per_image[img_file_name] = pred_file_for_img

	return EvalFiles(
		custom_preds_dir=custom_preds_dir,
		custom_anns_dir=custom_anns_dir,
		coco_preds_file=coco_preds_file,
		coco_anns_file=coco_anns_file,
		coco_ann_file_per_image=coco_ann_file_per_image,
		coco_pred_file_per_image=coco_pred_file_per_image,
	)

def _evaluate_on_dataset(eval_files: EvalFiles, model_name: str):
	ground_truth = COCO(eval_files.coco_anns_file)
	detections = ground_truth.loadRes(str(eval_files.coco_preds_file))

	E = COCOeval(ground_truth, detections)
	E.evaluate()
	E.accumulate()
	E.summarize()
	mAP_on_dataset = round(float(E.stats[0]), 3)

	# basically the same code as the accumulate() function, with
	# slighly adaptations to get the intermediary results that I want:
	# 	n_tp, n_fp, n_fn for a given IoU (0.5), areaRng (all) and maxDets (100).

	# anything involving "_pe" is about parameters used on evalute()
	# (things that WERE computed)
	_pe = E._paramsEval
	setK = set(_pe.catIds)
	setA = set(map(tuple, _pe.areaRng))
	setM = set(_pe.maxDets)
	setI = set(_pe.imgIds)
	I0 = len(_pe.imgIds)
	A0 = len(_pe.areaRng)

	# anything involving "p" is about the parameters I want now
	# (things that WILL BE taken into account to calculate the mAP)
	p = E.params
	p.maxDets = [100]
	p.areaRng = [p.areaRng[0]] # all
	k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
	m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
	a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
	i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]

	n_tp_on_dataset = 0
	n_fp_on_dataset = 0
	n_fn_on_dataset = 0
	for k, k0 in enumerate(k_list):
		Nk = k0*A0*I0
		# since I'm only interested in one areaRng and one maxDet,
		# there's no need for those loops
		a, a0 = 0, a_list[0]
		Na = a0*I0
		m, maxDet = 0, m_list[0]

		# be very careful here, since I'm on a different scope
		# "self" on their code is "E" here
		# "E" on their code is what I called "filtered_evalImgs" here
		filtered_evalImgs = [E.evalImgs[Nk + Na + i] for i in i_list]
		filtered_evalImgs = [e for e in filtered_evalImgs if not e is None]
		if len(filtered_evalImgs) == 0:
			continue
		dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in filtered_evalImgs])

		inds = np.argsort(-dtScores, kind='mergesort')
		dtScoresSorted = dtScores[inds]

		# each "e" in evalImgs contains info that refers to one category in
		# one image, with one areaRng (but this last one is irrelevant here).
		# 
		# each e['dtMatches'] is a list of lists, where each line / inner list
		# refers to one IoU threshold (10 thresholds by default, so 10 lines) and
		# each column / i-element of each list is a detection id. For each IoU
		# and detection id, we have the ground truth id that best matches that
		# detection, or 0 if there was no match.
		# for instance "e['dtMatches'][0,1] = 2" means that at the first IoU
		# threshold (index 0), the detection with id 1 was matched to the
		# ground truth with id 2
		#
		# they concatenate all of those, per IoU, because it doesn't matter 
		# from which image it is or what category it is: all that matters is
		# whether or not there was a match
		dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in filtered_evalImgs], axis=1)[:,inds]
		dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in filtered_evalImgs], axis=1)[:,inds]
		tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
		fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

		# for each threshold, they compute the cum_sum of tp and fp
		# (aparently it has to do with the way precision and recall is
		# calculated, I had no idea)
		# tp_sum[0] is the cum_sum_tp for the first threshold
		# tp_sum[1] is the cum_sum_tp for the second threshold and so on
		# tp_sum[tind,x] is the n_tp in the first x detections
		# tp_sum[tind,-1], which I use below, is the n_tp for that category
		tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
		fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)

		# I'm counting tp/fp/fn for IoU 0.5, which is the first one,
		# and I only care about the "total" tp/fp, which is the last item of the sum
		n_tp_for_that_k = tp_sum[0][-1]
		n_fp_for_That_k = fp_sum[0][-1]

		# Now for false negatives...
		# Same logic applies to e['gtMatches']
		gtm = np.concatenate([e['gtMatches'] for e in filtered_evalImgs], axis=1)
		n_fns_per_threshold = np.apply_along_axis(lambda x: np.array(np.where(x == 0)).size, 1, gtm)
		 # 0 means no match
		my_n_fn = n_fns_per_threshold[0]

		# I could also use the values they compute. Since they use
		# "recall = tp / npig", npig must be "TP + FN" (that's literally the
		# definition of recall). Since I have the n_tp, all I gotta go
		# is subtract it
		gtIg = np.concatenate([e['gtIgnore'] for e in filtered_evalImgs])
		npig = np.count_nonzero(gtIg==0 )
		if npig == 0:
			continue
		their_n_fn = npig - tp_sum[0][-1]

		if my_n_fn != their_n_fn:
			print("I messed up on the fn computation")
		n_fn_for_that_k = my_n_fn

		n_tp_on_dataset += int(n_tp_for_that_k)
		n_fp_on_dataset += int(n_fp_for_That_k)
		n_fn_on_dataset += int(n_fn_for_that_k)

	n_anns_considered = Annotations(eval_files.custom_anns_dir).get_n_objects()
	n_preds_considered = Predictions(eval_files.custom_preds_dir).get_n_objects(model_name)
	
	return DatasetResults(
		n_anns_considered=n_anns_considered,
		n_preds_considered=n_preds_considered,
		mAP=mAP_on_dataset,
		n_true_positives=n_tp_on_dataset,
		n_false_positives=n_fp_on_dataset,
		n_false_negatives=n_fn_on_dataset,
	)

def _evaluate_per_image(eval_files: EvalFiles, model_name: str):
	coco_ann_manager = COCOAnnotations(eval_files.coco_anns_file)
	img_map = coco_ann_manager.img_map()
	classmap = coco_ann_manager.classmap_by_id()
	results_per_img = {}

	for img_name, img_id in img_map.items():
		# Tecnicamente eu poderia só usar a API setando E.params.imgIds, mas
		# eu prefiro "mentir" para API com um dataset de 1 imagem do que
		# depender dessa funcionalidade deles (vai saber o que muda na hora
		# de armazenar os resultados)
		ann_file = eval_files.coco_ann_file_per_image[img_name]
		pred_file = eval_files.coco_pred_file_per_image[img_name]
		ground_truth = COCO(ann_file)
		detections = ground_truth.loadRes(str(pred_file))
		E = COCOeval(ground_truth, detections)
		E.evaluate()
		E.accumulate()
		E.summarize()
		mAP_on_img = round(float(E.stats[0]), 3)

		# Same thing I did in evaluate_dataset...
		_pe = E._paramsEval
		setK = set(_pe.catIds)
		setA = set(map(tuple, _pe.areaRng))
		setM = set(_pe.maxDets)
		setI = set(_pe.imgIds)
		I0 = len(_pe.imgIds)
		A0 = len(_pe.areaRng)

		p = E.params
		p.maxDets = [100]
		p.areaRng = [p.areaRng[0]] # all
		k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
		m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
		a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
		i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]

		# ...except here, where I want the actual preds and anns instead of just
		# the number. It's still mostly similar, with the changes accompanied
		# by comments
		tps_on_img = []
		fps_on_img = []
		fns_on_img = []
		for k, k0 in enumerate(k_list):
			Nk = k0*A0*I0
			a, a0 = 0, a_list[0]
			Na = a0*I0
			m, maxDet = 0, m_list[0]
			
			# It's only one image now, no need to loop
			evalImg_for_that_k = E.evalImgs[Nk + Na + i_list[0]]

			dtScores = np.array(evalImg_for_that_k['dtScores'][0:maxDet])

			inds = np.argsort(-dtScores, kind='mergesort')
			dtScoresSorted = dtScores[inds]

			# Get dts and gts for that image and category
			img_id = evalImg_for_that_k['image_id']
			cat_id = evalImg_for_that_k['category_id']
			dts = E._dts[img_id, cat_id]
			gts = E._gts[img_id, cat_id]
			dtIds = evalImg_for_that_k['dtIds']
			gtIds = evalImg_for_that_k['gtIds']

			dtm_per_threshold = evalImg_for_that_k['dtMatches'][:,0:maxDet][:,inds]
			# I only want dtm for IoU = 0.5, so index 0
			for dtId, gtId in zip(dtIds, dtm_per_threshold[0]):
				actual_dt = next(dt for dt in dts if dt['id'] == dtId)
				dt_in_custom_format = Predictions.from_coco_format(actual_dt, classmap)
				if gtId == 0:
					# false positive
					fps_on_img.append(dt_in_custom_format)
				else:
					# true positive					
					actual_gt = next(gt for gt in gts if gt['id'] == int(gtId)) # not sure why gtId is a float
					gt_in_custom_format = Annotations.from_coco_format(actual_gt, classmap)
					tps_on_img.append((dt_in_custom_format, gt_in_custom_format))
			
			gtm_per_threshold = evalImg_for_that_k['gtMatches']
			# I only want gtm for IoU = 0.5, so index 0
			for gtId, dtId in zip(gtIds, gtm_per_threshold[0]):
				actual_gt = next(gt for gt in gts if gt['id'] == int(gtId)) # again, not sure why it's a float
				gt_in_custom_format = Annotations.from_coco_format(actual_gt, classmap)
				if dtId == 0:
					# false negative
					fns_on_img.append(actual_gt)

		n_anns_considered = Annotations(eval_files.coco_ann_file_per_image[img_name]). \
		                    get_n_objects()
		n_preds_considered = Predictions(eval_files.coco_pred_file_per_image[img_name]). \
		                     get_n_objects(model_name)

		results_per_img[img_name] = ImageResults(
			n_anns_considered=n_anns_considered,
			n_preds_considered=n_preds_considered,
			mAP=mAP_on_img,
			true_positives=tps_on_img,
			false_positives=fps_on_img,
			false_negatives=fns_on_img,
		)
	
	return results_per_img

def _save_results(results: ModelResults, out_dir: Path):
	results_dict = asdict(results)

	results_file = out_dir / 'results.json'
	with results_file.open('w') as f:
		json.dump(results_dict, f)