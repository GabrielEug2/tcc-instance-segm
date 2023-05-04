from pathlib import Path

from tqdm import tqdm

from personal_lib.core import Predictions
from . import plot_lib

def plot(pred_dir: Path, img_dir: Path, out_dir: Path):
	model_names = (d.stem for d in pred_dir.glob('*/'))
	for model_name in model_names:
		pred_manager = Predictions(pred_dir / model_name)

		img_files = img_dir.glob('*.jpg')
		for img_file in tqdm(img_files):
			predictions = pred_manager.load(img_file.stem)

			predictions_img_file = out_dir / img_file.stem / f"{model_name}.jpg"
			predictions_img_file.parent.mkdir(parents=True, exist_ok=True)
			plot_lib.plot(predictions, img_file, predictions_img_file)

			mask_out_dir = out_dir / img_file.stem / f"{model_name}_masks"
			mask_out_dir.mkdir(parents=True, exist_ok=True)
			plot_lib.plot_individual_masks(predictions, mask_out_dir, img_file)