from pathlib import Path

from personal_lib import pred_parser

from . import common_logic

def plot(pred_dir: Path, img_files: list[Path], save_masks: bool):
	for img_file in img_files:
		pred_files = pred_dir.glob(f"{img_file.stem}_*.json")

		for pred_file in pred_files:
			model_name = pred_file.stem.split('_')[1]
			predictions = pred_parser.load_preds(pred_file)

			predictions_img_file = pred_dir / f"{img_file.stem}_{model_name}.jpg"
			common_logic.plot(predictions, img_file, predictions_img_file)

			if save_masks:
				mask_out_dir = pred_dir / f"{img_file.stem}_{model_name}_masks"
				common_logic.plot_individual_masks(predictions, mask_out_dir, img_file)