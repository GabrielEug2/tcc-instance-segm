from enum import Enum
from pathlib import Path

from segm_lib.core.structures import Annotation, Prediction
from segm_lib.plot.detectron_plot_lib import DetectronPlotLib

from ..structures.eval_results import EvalResults

detectron_plot_lib = DetectronPlotLib()

def plot_tps_fps_fns(img_results: EvalResults, out_file: Path, img_file: Path):
	obj_list = _make_obj_list_to_plot(img_results)
	colors = _assign_colors(obj_list)
	detectron_plot_lib.plot(obj_list, img_file, out_file, colors=colors)

def _make_obj_list_to_plot(img_results: EvalResults) -> list[Annotation|Prediction]:
	objs_to_plot = []
	for classname, tp_list in img_results.true_positives['list_per_class'].items():
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

	for classname, fp_list in img_results.false_positives['list_per_class'].items():
		classname_for_plot = f'{classname}_FP'
		for fp in fp_list:
			confidence = fp.confidence
			mask = fp.mask
			bbox = fp.bbox
			objs_to_plot.append(Prediction(classname_for_plot, confidence, mask, bbox))

	for classname, fn_list in img_results.false_negatives['list_per_class'].items():
		classname_for_plot = f'{classname}_FN'
		for fn in fn_list:
			mask = fn.mask
			bbox = fn.bbox
			objs_to_plot.append(Annotation(classname_for_plot, mask, bbox))

	return objs_to_plot

class Colors(Enum):
	GREEN = (11, 181, 85)
	GREY = (140, 140, 140)
	BLUE = (44, 144, 232)
	RED = (230, 25, 25)
	
def _assign_colors(objs: list[Annotation|Prediction]) -> list[Colors]:
	colors = {}

	class_list = (o.classname for o in objs)
	for classname in class_list:
		if classname.endswith('_TP_det'):
			colors[classname] = Colors.GREEN.value
		elif classname.endswith('_TP_ann'):
			colors[classname] = Colors.GREY.value
		elif classname.endswith('_FP'):
			colors[classname] = Colors.BLUE.value
		else: # _FN
			colors[classname] = Colors.RED.value

	return colors