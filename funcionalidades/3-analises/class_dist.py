import argparse
import json
from pathlib import Path

from personal_lib.parsing.predictions import PredictionManager

parser = argparse.ArgumentParser()
parser.add_argument('pred_dir', help='Directory where you placed the annotations')
args = parser.parse_args()

pred_dir = Path(args.pred_dir)
pred_manager = PredictionManager(pred_dir)

model_names = pred_manager.get_model_names()
for model_name in model_names:
	class_dist = pred_manager.class_distribution(model_name)

	out_file = Path(__file__).parent / f"classdist_{model_name}.json"
	sorted_dist = dict(sorted(class_dist.items(), key=lambda c: c[1], reverse=True))
	with out_file.open('w') as f:
		json.dump(sorted_dist, f, indent=4)