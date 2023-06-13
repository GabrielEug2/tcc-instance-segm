from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from segm_lib.core.structures import Annotation, Prediction
from segm_lib.core.mask_conversions import ann_to_rle

class APIResults:	
	def __init__(self, detailed: bool):
		self.AP = 0.0

		# just for type hinting, the actual initialization is below
		self.true_positives = {}
		self.false_positives = {}
		self.false_negatives = {}

		for attrname in ['true_positives', 'false_positives', 'false_negatives']:
			setattr(self, attrname, {
				'n': 0,
				'n_per_class': {},
			})

			if detailed:
				getattr(self, attrname)['list_per_class'] = {}

def eval(coco_anns_file: Path, coco_preds_file: Path, detailed=False) -> APIResults:
	"""Evaluates the data.

	Args:
		coco_anns_file (Path): file containing coco-formatted annotations.
		coco_preds_file (Path): file containing coco-formatted predictions.
		detailed (bool, optional): if True, returns extensive info about
		 	the data. Use only when the anns and preds refer to *one* image.

	Returns:
		APIResults object with following fields:
			AP: float
			true_positives / false_positives / false_negatives: dict with keys:
				n: int
				n_per_class: dict[str, int]
				(if detailed) list_per_class: dict[str, list[things]] (see below)
					* tuple[pred,ann] for true positives
					* preds for false positives
					* anns for false negatives
	"""
	ground_truth = COCO(coco_anns_file)
	try:
		detections = ground_truth.loadRes(str(coco_preds_file))
	except IndexError:
		# no predictions
		# every ground_truth is a false negative
		n_fns_per_class = defaultdict(lambda: 0)
		if detailed:
			list_fns_per_class = defaultdict(list)

		for gt_in_coco_format in ground_truth.anns.values():
			classname = ground_truth.cats[gt_in_coco_format['category_id']]['name']

			img_id = gt_in_coco_format['image_id']
			h, w = ground_truth.imgs[img_id]['height'], ground_truth.imgs[img_id]['width']
			gt_in_coco_format['segmentation'] = ann_to_rle(gt_in_coco_format['segmentation'], h, w)
			gt_in_custom_format = Annotation.from_coco_format(gt_in_coco_format, classname)

			n_fns_per_class[classname] += 1
			if detailed:
				list_fns_per_class[classname].append(
					gt_in_custom_format
				)

		n_fn = sum(n_fns_per_class.values())

		results = APIResults(detailed)
		results.false_negatives['n'] = n_fn
		results.false_negatives['n_per_class'] = dict(n_fns_per_class)
		if detailed:
			results.false_negatives['list_per_class'] = dict(list_fns_per_class)

		return results

	E = COCOeval(ground_truth, detections)
	E.evaluate()
	E.accumulate()
	E.summarize()
	average_precision = round(float(E.stats[0]), 3)

	# basically the same code as the accumulate() function, with slighly
	# adaptations to get what I want (true positives, false positives and
	# false negatives for a given IoU (0.5), areaRng (all) and maxDets (100)).

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
	# (things that WILL BE taken into account to calculate the AP)
	p = E.params
	p.maxDets = [100]
	p.areaRng = [p.areaRng[0]] # all
	k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
	# I'll also want the catId for each value in k_list, so:
	cat_ids_from_k = [k for n, k in enumerate(p.catIds) if k in setK]
	m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
	a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
	i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]

	n_tps_per_class = {}
	n_fps_per_class = {}
	n_fns_per_class = {}
	list_tps_per_class = defaultdict(list)
	list_fps_per_class = defaultdict(list)
	list_fns_per_class = defaultdict(list)
	for k, k0 in enumerate(k_list):
		# get classname from this k
		cat_id = cat_ids_from_k[k]
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
			# no annotations nor predictions for that k
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
		dtm = np.concatenate([e['dtMatches'][:,0:maxDet] for e in evalImgs_for_that_k], axis=1)[:,inds]

		# They calculate the number of tp/fp/fn slightly diferently, because
		# of the way AP is calculated, but I don't need to do that
		n_tps_per_threshold = np.count_nonzero(dtm != 0, axis=1)
		n_fps_per_threshold = np.count_nonzero(dtm == 0, axis=1)
		# I'm only counting tp/fp/fn for IoU 0.5, which is the first one
		n_tps_per_class[classname] = int(n_tps_per_threshold[0])
		n_fps_per_class[classname] = int(n_fps_per_threshold[0])
		
		# Now for false negatives...
		# Same logic I explained for e['dtMatches'] applies to e['gtMatches'].
		# "e['gtMatches'][0,1] = 2" means that, at the first IoU threshold
		# (index 0), the first ground truth of that img and category was
		# matched to the detection with id 2.
		# concatenated because, for the calculations, it doesn't matter from
		# which img it is
		gtm = np.concatenate([e['gtMatches'] for e in evalImgs_for_that_k], axis=1)
		n_fns_per_threshold = np.count_nonzero(gtm == 0, axis=1)
		# again, I'm only interested on IoU 0.5, which is the first one
		n_fns_per_class[classname] = int(n_fns_per_threshold[0])

		if detailed:
			# To get an actual list of tps/fps/fns, it's a bit more complicated.
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
						list_fps_per_class[classname].append(
							dt_in_custom_format
						)
					else:
						# true positive
						gt_in_coco_format = ground_truth.anns[gtId]
						_fix_rle(gt_in_coco_format)
						gt_in_custom_format = Annotation.from_coco_format(gt_in_coco_format, classname)

						list_tps_per_class[classname].append(
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
						list_fns_per_class[classname].append(
							gt_in_custom_format
						)
					# else:
						# true positive, already saved

	results = APIResults(detailed)
	results.AP = average_precision
	for attrname, n_per_class, list_per_class in [
			['true_positives', n_tps_per_class, list_tps_per_class],
			['false_positives', n_fps_per_class, list_fps_per_class],
			['false_negatives', n_fns_per_class, list_fns_per_class],
			]:
		n = sum(n_per_class.values())

		getattr(results, attrname)['n'] = n
		getattr(results, attrname)['n_per_class'] = n_per_class
		if detailed:
			getattr(results, attrname)['list_per_class'] = dict(list_per_class)

	return results

def _fix_rle(gt):
	# Not sure why sometimes it's bytes and sometimes it's str
	if type(gt['segmentation']['counts']) == bytes:
		gt['segmentation']['counts'] = gt['segmentation']['counts'].decode('utf-8')