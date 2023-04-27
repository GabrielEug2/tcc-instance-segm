import json
from pathlib import Path

from personal_lib.parsing.annotations import AnnotationManager
from personal_lib.parsing.predictions import PredictionManager

def class_dist(ann_file: Path, pred_dir: Path, out_dir: Path):
	class_dist_out_dir = out_dir / 'class_dist'
	class_dist_out_dir.mkdir(parents=True, exist_ok=True)

	_class_dist_for_anns(ann_file, class_dist_out_dir)
	_class_dist_for_preds(pred_dir, class_dist_out_dir)

def _class_dist_for_anns(ann_file: Path, out_dir: Path):
	anns = AnnotationManager(ann_file)
	class_dist = anns.class_distribution()

	out_file = out_dir / "groundtruth.json"
	dist_sorted_by_count = dict(sorted(class_dist.items(), key=lambda c: c[1], reverse=True))
	with out_file.open('w') as f:
		json.dump(dist_sorted_by_count, f, indent=4)

def _class_dist_for_preds(pred_dir: Path, out_dir: Path):
	pred_manager = PredictionManager(pred_dir)
	model_names = pred_manager.get_model_names()
	for model_name in model_names:
		class_dist = pred_manager.class_distribution(model_name)

		out_file = out_dir / f"{model_name}.json"
		dist_sorted_by_count = dict(sorted(class_dist.items(), key=lambda c: c[1], reverse=True))
		with out_file.open('w') as f:
			json.dump(dist_sorted_by_count, f, indent=4)