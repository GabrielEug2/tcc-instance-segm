import json
from pathlib import Path

from personal_lib.parsing.annotations import AnnotationManager
from personal_lib.parsing.predictions import PredictionManager
from personal_lib.parsing.common.files import save_class_dist

def class_dist(pred_dir: Path, ann_file: Path, out_dir: Path) -> dict:
	_class_dist_for_anns(ann_file, out_dir)
	_class_dist_for_preds(pred_dir, out_dir)

def _class_dist_for_anns(ann_file: Path, out_dir: Path):
	anns = AnnotationManager(ann_file)
	class_dist = anns.class_distribution()

	save_class_dist(class_dist, out_dir / 'groundtruth.json')

def _class_dist_for_preds(pred_dir: Path, out_dir: Path):
	pred_manager = PredictionManager(pred_dir)
	model_names = pred_manager.get_model_names()

	for model_name in model_names:
		class_dist = pred_manager.class_distribution(model_name)

		save_class_dist(class_dist, out_dir / f"{model_name}.json")