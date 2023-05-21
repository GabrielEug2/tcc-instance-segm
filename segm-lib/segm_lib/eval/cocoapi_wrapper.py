from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..core.structures import Annotation, Prediction
from .structures.model_results import TP_FP_FN_ShortInfo, TP_FP_FN_DetailedInfo


@dataclass
class APIResults:
	mAP: float = 0.0
	true_positives: TP_FP_FN_ShortInfo|TP_FP_FN_DetailedInfo = None
	false_positives: TP_FP_FN_ShortInfo|TP_FP_FN_DetailedInfo = None
	false_negatives: TP_FP_FN_ShortInfo|TP_FP_FN_DetailedInfo = None

def eval(coco_anns_file: Path, coco_preds_file: Path, detailed=False) -> APIResults:
	"""Computes mAP and tp/fp/fn info for the data.

	Args:
		coco_anns_file (Path): file containing coco-formatted annotations.
		coco_preds_file (Path): file containing coco-formatted predictions.
		detailed (bool, optional): if False, returns the number of tp/fp/fn.
			If True, returns a list of tp/fp/fn for each class. Defaults to
			False.

	Returns:
		APIResults: has the following fields:
			mAP: float
			true_positives / false_positives / false_negatives: TP_FP_FN_ShortInfo
			 	if detailed is False, TP_FP_FN_DetailedInfo if detailed is True
	"""
	results = APIResults()

	ground_truth = COCO(coco_anns_file)
	detections = ground_truth.loadRes(str(coco_preds_file))

	E = COCOeval(ground_truth, detections)
	E.evaluate()
	E.accumulate()
	E.summarize()
	results.mAP = round(float(E.stats[0]), 3)

	# basically the same code as the accumulate() function, with slighly
	# adaptations to get what I want (number of tp, fp and fn for a given
	# IoU (0.5), areaRng (all) and maxDets (100) for simple_eval and the
	# actual list of tp, fp and fn for detailed_eval.

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
	true_positives_per_class = defaultdict(list)
	false_positives_per_class = defaultdict(list)
	false_negatives_per_class = defaultdict(list)
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
		# each column / i-th element of each list represents a detection. For each
		# IoU and detection, we have the ground truth id that best matches that
		# detection, or 0 if there was no match.
		# for instance "e['dtMatches'][0,1] = 2" means that at the first IoU
		# threshold (index 0), the second detection for that img and category
		# (index 1) was matched to the ground truth with id 2.
		#
		# they concatenate all of those, per IoU, because for the tp/fp/fn
		# computations, it doesn't matter from which image it is: all that
		# matters is whether or not there was a match.
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
		# (index 0), the first ground truth of that img and category was
		# matched to the detection with id 2.
		# concatenated because, for the calculations, it doesn't matter from
		# which img it is
		gtm = np.concatenate([e['gtMatches'] for e in evalImgs_for_that_k], axis=1)
		n_fns_per_threshold = np.count_nonzero(gtm == 0, axis=1)
		n_fn = n_fns_per_threshold[0]
		# I could also just use the values they compute. They compute recall as
		# "recall = tp / npig", so npig *must be* "TP + FN" (that's literally the
		# definition of recall). Since I have the n_tp, all I gotta do is subtract
		# it.
		# gtIg = np.concatenate([e['gtIgnore'] for e in evalImgs_for_that_k])
		# npig = np.count_nonzero(gtIg==0 )
		# if npig == 0:
		# 	continue
		# n_fn = npig - tp_sum[0][-1]
		n_false_negatives_per_class[classname] = int(n_fn)

		if detailed:
			# To get an actual list of tps/fps/fns, it's a bit more complicated:
			#
			# We can't concatenate, because I will need those detection indexes
			# and ground truth indexes to find the actual detections and ground
			# truths
			for evalImg in evalImgs_for_that_k:
				dtScores = np.array(evalImg['dtScores'][0:maxDet])
				inds = np.argsort(-dtScores, kind='mergesort')

				# To get true positives and false positives...
				dtm_per_threshold = evalImg['dtMatches'][:,0:maxDet][:,inds]

				# I only want dtm for IoU = 0.5, so index 0
				for dtIndex, gtId in enumerate(dtm_per_threshold[0]):
					dtId = evalImg['dtIds'][dtIndex]

					dt_in_coco_format = detections.anns[dtId]
					dt_in_custom_format = Prediction.from_coco_format(dt_in_coco_format, classname)

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
						gt_in_custom_format = Annotation.from_coco_format(gt_in_coco_format, classname)

						true_positives_per_class[classname].append(
							(dt_in_custom_format, gt_in_custom_format)
						)

				# To get false negatives...
				gtm_per_threshold = evalImg['gtMatches']

				# I only want gtm for IoU = 0.5, so index 0
				for gtIndex, dtId in enumerate(gtm_per_threshold[0]):
					gtId = evalImg['gtIds'][gtIndex]

					gt_in_coco_format = ground_truth.anns[gtId]
					_fix_rle(gt_in_coco_format)
					gt_in_custom_format = Annotation.from_coco_format(gt_in_coco_format, classname)

					dtId = int(dtId) # again, not sure why they store matched ids as floats
					if dtId == 0:
						# false negative
						false_negatives_per_class[classname].append(
							gt_in_custom_format
						)
					# else:
						# true positive, already saved

	computed_values = {
		'true_positives': (n_true_positives_per_class, true_positives_per_class),
		'false_positives': (n_false_positives_per_class, false_positives_per_class),
		'false_negatives': (n_false_negatives_per_class, false_negatives_per_class)
	}
	for metric_name, (n_per_class, list_per_class) in computed_values.items():
		n = sum(n_per_class.values())

		if detailed:
			metric_values = TP_FP_FN_DetailedInfo(
				n=n,
				n_per_class=n_per_class,
				list_per_class=dict(list_per_class)
			)
		else:
			metric_values = TP_FP_FN_ShortInfo(
				n=n,
				n_per_class=n_per_class,
			)

		if not hasattr(results, metric_name):
			raise ValueError((f'Implementation error. Object of class '
		                      f'{results.__class__.__name__} does not '
							  f'have a "{metric_name}" field.'))
		setattr(results, metric_name, metric_values)

	return results

def _fix_rle(gt):
	# Not sure why sometimes it's bytes and sometimes it's str
	if type(gt['segmentation']['counts']) == bytes:
		gt['segmentation']['counts'] = gt['segmentation']['counts'].decode('utf-8')