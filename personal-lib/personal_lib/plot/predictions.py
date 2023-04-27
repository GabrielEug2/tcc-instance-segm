from pathlib import Path

from tqdm import tqdm

from personal_lib.parsing.predictions import PredictionManager
from . import plot_lib

def plot(pred_dir: Path, img_dir: Path, out_dir: Path):
	pred_manager = PredictionManager(pred_dir)
	model_names = pred_manager.get_model_names()
	img_files = img_dir.glob('*.jpg')

	for img_file in tqdm(img_files):
		for model_name in model_names:
			predictions = pred_manager.load(img_file.stem, model_name)

			predictions_img_file = out_dir / img_file.stem / f"{model_name}.jpg"
			predictions_img_file.parent.mkdir(parents=True, exist_ok=True)
			plot_lib.plot(predictions, img_file, predictions_img_file)

			mask_out_dir = out_dir / img_file.stem / f"{model_name}_masks"
			mask_out_dir.mkdir(parents=True, exist_ok=True)
			plot_lib.plot_individual_masks(predictions, mask_out_dir, img_file)