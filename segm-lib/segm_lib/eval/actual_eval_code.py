from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from segm_lib.ann_manager import AnnManager
from segm_lib.pred_manager import PredManager
from segm_lib.eval.structures.model_results import *
from segm_lib.coco_annotations import COCOAnnotations
from segm_lib.structures import Annotation, Prediction

@dataclass
class EvalFilesForImg:
	custom_anns_dir: Path = None
	custom_preds_dir: Path = None
	coco_anns_file: Path = None
	coco_preds_file: Path = None

@dataclass
class EvalFiles:
	custom_anns_dir: Path = None
	custom_preds_dir: Path = None
	coco_anns_file: Path = None
	coco_preds_file: Path = None
	per_image: dict[str, EvalFilesForImg] = None

def prep_eval_files(
	model_name: str,
	pred_manager: PredManager,
	ann_manager: AnnManager,
	original_ann_file: Path,
	common_classes: list[str],
	out_dir: Path
):	
	custom_anns_dir = out_dir / 'custom_anns'
	ann_manager.filter(custom_anns_dir, classes=common_classes)

	custom_preds_dir = out_dir / 'custom_preds'
	pred_manager.filter(custom_preds_dir, model_name, classes=common_classes)

	# Ideally I would build the ann file from my own annotation format, but the
	# eval API uses some fields which I don't save on mine, like iscrowd. It's
	# easier to just filter the original file too than modifying my format
	# to include all those fields.
	coco_anns_file = out_dir / 'coco_anns.json'
	COCOAnnotations(original_ann_file).filter(coco_anns_file, classes=common_classes)

	coco_preds_file = out_dir / 'coco_preds.json'
	filtered_coco_anns_manager = COCOAnnotations(coco_anns_file)
	classmap = filtered_coco_anns_manager.classmap()
	img_map = filtered_coco_anns_manager.img_map()
	filtered_pred_manager = PredManager(custom_preds_dir)
	filtered_pred_manager.to_coco_format(model_name, img_map, classmap, coco_preds_file)

	eval_files_per_img = {}
	for img_file_name, img_id in img_map.items():
		base_path = out_dir / 'per_image' / img_file_name

		custom_ann_dir_for_img = base_path / "custom_anns"
		AnnManager(custom_anns_dir).filter(custom_ann_dir_for_img, img_file_name=img_file_name)

		custom_pred_dir_for_img = base_path / "custom_preds"
		PredManager(custom_preds_dir).filter(custom_pred_dir_for_img, model_name, img_file_name=img_file_name)

		coco_ann_file_for_img = base_path / "coco_anns.json"
		filtered_coco_anns_manager.filter(coco_ann_file_for_img, img_file_name=img_file_name)

		coco_pred_file_for_img = base_path / "coco_preds.json"
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
	
def evaluate_on_dataset(eval_files: EvalFiles, model_name: str):
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

	ground_truth = COCO(eval_files.coco_anns_file)
	detections = ground_truth.loadRes(str(eval_files.coco_preds_file))
	
	E = COCOeval(ground_truth, detections)
	E.evaluate()
	E.accumulate()
	E.summarize()
	mAP_on_dataset = round(float(E.stats[0]), 3)
	dataset_results.mAP = mAP_on_dataset

	# basically the same code as the accumulate() function, with slighly
	# adaptations to get what I want (number of tp, fp and fn for a given
	# IoU (0.5), areaRng (all) and maxDets (100).

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

	# Since nothing here is simple, 'k0' isn't the actual cat id, just
	# the index for it in k_list. To get the actual id:
	cat_ids_from_k0 = [k for n, k in enumerate(p.catIds) if k in setK]

	n_true_positives_per_class = {}
	n_false_positives_per_class = {}
	n_false_negatives_per_class = {}
	for k, k0 in enumerate(k_list):
		# get cat_name from this k0
		cat_id = cat_ids_from_k0[k0]
		classname = ground_truth.cats[cat_id]['name']

		Nk = k0*A0*I0
		# since I'm only interested in one areaRng and one maxDet,
		# there's no need for those loops
		a, a0 = 0, a_list[0]
		Na = a0*I0
		m, maxDet = 0, m_list[0]

		# be very careful here, since I'm on a different scope
		# "self" on their code is "E" here
		# "E" on their code is what I called "evalImgs_for_that_k" here
		evalImgs_for_that_k = [E.evalImgs[Nk + Na + i] for i in i_list]
		evalImgs_for_that_k = [e for e in evalImgs_for_that_k if not e is None]
		if len(evalImgs_for_that_k) == 0:
			continue
		dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in evalImgs_for_that_k])

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
		# ground truth with id 2 (is not *quite* like that, as I noticed when
		# I implemented the evaluate_images(), but that's a decent enough
		# explanation if you just want the number of tps/fps/fns)
		#
		# they concatenate all of those, per IoU, because it doesn't matter 
		# from which image it is or what category it is: all that matters is
		# whether or not there was a match
		dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in evalImgs_for_that_k], axis=1)[:,inds]
		dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in evalImgs_for_that_k], axis=1)[:,inds]
		tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
		fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

		# for each threshold, they compute the cum_sum of tp and fp
		# (aparently it has to do with the way precision and recall is
		# calculated, I had no idea)
		# tp_sum[0] is the cum_sum_tp for the first threshold
		# tp_sum[1] is the cum_sum_tp for the second threshold
		# and so on
		# tp_sum[tind,x] is the n_tp in the first x detections
		# tp_sum[tind,-1], which I use below, is the n_tp for that category
		tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
		fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)

		# I'm counting tp/fp/fn for IoU 0.5, which is the first one,
		# and I only care about the "total" tp/fp, which is the last item of the sum
		n_true_positives_per_class[classname] = int(tp_sum[0][-1])
		n_false_positives_per_class[classname] = int(fp_sum[0][-1])

		# Now for false negatives...
		# Same logic I explained for e['dtMatches'] applies to e['gtMatches'].
		# "e['gtMatches'][0,1] = 2" means that, at the first IoU threshold
		# (index 0), the ground truth with id 1 was matched to the detection
		# with id 2 (again, it's not *quite* like that, as I will explain in
		# evaluate_images(), but that's a decent enough description if you
		# just want the number of tps/fps/fns)
		gtm = np.concatenate([e['gtMatches'] for e in evalImgs_for_that_k], axis=1)
		n_fns_per_threshold = np.count_nonzero(gtm == 0, axis=1)
		n_fn = n_fns_per_threshold[0]

		# I could also use the values they compute. Since they use
		# "recall = tp / npig", npig must be "TP + FN" (that's literally the
		# definition of recall). Since I have the n_tp, all I gotta go
		# is subtract it
		# gtIg = np.concatenate([e['gtIgnore'] for e in evalImgs_for_that_k])
		# npig = np.count_nonzero(gtIg==0 )
		# if npig == 0:
		# 	continue
		# n_fn = npig - tp_sum[0][-1]

		n_false_negatives_per_class[classname] = int(n_fn)
	
	metrics_per_class = {
		'true_positives': n_true_positives_per_class,
		'false_positives': n_false_positives_per_class,
		'false_negatives': n_false_negatives_per_class
	}
	for metric_name, n_per_class in metrics_per_class.items():
		n = sum(n_per_class.values())

		if not hasattr(dataset_results, metric_name):
			raise ValueError((f"Implementation error. Object of class "
		                      f"{dataset_results.__class__.__name__} "
			                  f"doesn't have a {metric_name} field."))
		setattr(
			dataset_results,
			metric_name,
			TP_FP_FN_ShortInfo(
				n=n,
				n_per_class=n_per_class,
			)
		)

	return dataset_results

def evaluate_per_image(eval_files: EvalFiles, model_name: str):
	coco_ann_manager = COCOAnnotations(eval_files.coco_anns_file)
	img_map = coco_ann_manager.img_map()
	classmap = coco_ann_manager.classmap_by_id()
	results_per_img = {}

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

		# Tecnicamente eu poderia só usar a API setando E.params.imgIds, mas
		# eu prefiro "mentir" para API com um dataset de 1 imagem do que
		# depender dessa funcionalidade deles (vai saber o que muda na hora
		# de armazenar os resultados)
		ann_file = eval_files.per_image[img_name].coco_anns_file
		pred_file = eval_files.per_image[img_name].coco_preds_file
		ground_truth = COCO(ann_file)
		detections = ground_truth.loadRes(str(pred_file))

		E = COCOeval(ground_truth, detections)
		E.evaluate()
		E.accumulate()
		E.summarize()
		mAP_on_img = round(float(E.stats[0]), 3)
		img_results.mAP = mAP_on_img

		# Same thing I did in evaluate_dataset()...
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

		cat_ids_from_k0 = [k for n, k in enumerate(p.catIds) if k in setK]

		# ...except here, where I want the actual preds and anns that are tp/fp/fn
		# instead of just the number. It's still mostly similar, with the changes
		# accompanied by comments.
		true_positives_per_class = defaultdict(list)
		false_positives_per_class = defaultdict(list)
		false_negatives_per_class = defaultdict(list)
		for k, k0 in enumerate(k_list):
			cat_id = cat_ids_from_k0[k0]
			classname = ground_truth.cats[cat_id]['name']
			
			Nk = k0*A0*I0
			a, a0 = 0, a_list[0]
			Na = a0*I0
			m, maxDet = 0, m_list[0]
			
			# It's only one image now, no need to loop
			evalImg_for_that_k = E.evalImgs[Nk + Na + i_list[0]]
			dtScores = np.array(evalImg_for_that_k['dtScores'][0:maxDet])

			inds = np.argsort(-dtScores, kind='mergesort')
			dtScoresSorted = dtScores[inds]

			# the detection id and ground truth id I mentioned in evaluate_dataset()
			# is actually a bit more complex than that.
			#
			# 	"e['dtMatches'][0,1] = 2" actually means that, at the first IoU
			# 	threshold (index 0), the ***2nd detection*** (index 1) for that img and category
			#	was matched to the ground truth with id 2.
			#
			# 	"e['gtMatches'][0,1] = 2" actually means that, at the first IoU
			# 	threshold (index 0), the ***2nd ground truth*** (index 1) for that img and category
			# 	was matches to the detection with id 2.
			#
			# The "2nd"s I described are not the actual ids to access those preds/anns,
			# just their index in the list of preds/anns for that specific img and
			# category. If you want the actual ids, you'll need to use those indexes on
			# e['dtIds'] and e['gtIds'].
			dtm_per_threshold = evalImg_for_that_k['dtMatches'][:,0:maxDet][:,inds]
			# I only want dtm for IoU = 0.5, so index 0
			for dtId, gtId in zip(evalImg_for_that_k['dtIds'], dtm_per_threshold[0]):
				dt_in_coco_format = detections.anns[dtId]
				dt_in_custom_format = Prediction.from_coco_format(dt_in_coco_format, classmap)

				gtId = int(gtId) # not sure why they store matched ids as floats
				if gtId == 0:
					# false positive
					false_positives_per_class[classname].append(
						dt_in_custom_format
					)
				else:
					# true positive
					gt_in_coco_format = ground_truth.anns[gtId]
					_fix_rle(gt_in_coco_format)
					gt_in_custom_format = Annotation.from_coco_format(gt_in_coco_format, classmap)

					true_positives_per_class[classname].append(
						(dt_in_custom_format, gt_in_custom_format)
					)
			
			gtm_per_threshold = evalImg_for_that_k['gtMatches']
			# I only want gtm for IoU = 0.5, so index 0
			for gtId, dtId in zip(evalImg_for_that_k['gtIds'], gtm_per_threshold[0]):
				gt_in_coco_format = ground_truth.anns[gtId]
				_fix_rle(gt_in_coco_format)
				gt_in_custom_format = Annotation.from_coco_format(gt_in_coco_format, classmap)

				dtId = int(dtId) # again, not sure why they store matched ids as floats
				if dtId == 0:
					# false negative
					false_negatives_per_class[classname].append(
						gt_in_custom_format
					)

		metrics_per_class = {
			'true_positives': true_positives_per_class,
			'false_positives': false_positives_per_class,
			'false_negatives': false_negatives_per_class
		}
		for metric_name, list_per_class in metrics_per_class.items():
			# defaultdict breaks the dataclasses.asdict() method I call later
			list_per_class = dict(list_per_class)
			n_per_class = {classname: len(list_) for classname, list_ in list_per_class.items()}
			n = sum(n_per_class.values())

			if not hasattr(img_results, metric_name):
				raise ValueError((f"Implementation error. Object of class "
								f"{img_results.__class__.__name__} "
								f"doesn't have a {metric_name} field."))
			setattr(
				img_results,
				metric_name,
				TP_FP_FN_DetailedInfo(
					n=n,
					n_per_class=n_per_class,
					list_per_class=list_per_class,
				)
			)

		results_per_img[img_name] = img_results
	
	return results_per_img

def _fix_rle(gt):
	# Not sure why sometimes it's bytes and sometimes it's str
	if type(gt['segmentation']['counts']) == bytes:
		gt['segmentation']['counts'] = gt['segmentation']['counts'].decode('utf-8')