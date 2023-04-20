from pathlib import Path

from personal_lib.parsing.predictions import Predictions
from . import plot_lib

def plot(pred_dir: Path, img_files: list[Path], save_masks: bool):
	for img_file in img_files:
		pred_files = pred_dir.glob(f"{img_file.stem}_*.json")

		for pred_file in pred_files:
			model_name = pred_file.stem.split('_')[1]
			predictions = Predictions.load_from_file(pred_file)

			predictions_img_file = pred_dir / f"{img_file.stem}_{model_name}.jpg"
			plot_lib.plot(predictions, img_file, predictions_img_file)

			if save_masks:
				mask_out_dir = pred_dir / f"{img_file.stem}_{model_name}_masks"
				plot_lib.plot_individual_masks(predictions, mask_out_dir, img_file)