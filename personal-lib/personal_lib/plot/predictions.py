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
			predictions = pred_manager.load(img_file, model_name)
			formatted_preds = _to_plot_format(predictions)

			predictions_img_file = out_dir / f"{img_file.stem}_{model_name}.jpg"
			plot_lib.plot(formatted_preds, img_file, predictions_img_file)

			mask_out_dir = out_dir / f"{img_file.stem}_{model_name}_masks"
			plot_lib.plot_individual_masks(formatted_preds, mask_out_dir, img_file)

def _to_plot_format(predictions):
	# Já está no formato certo, mas agora eu quero o dict
	# em si, não a classe em volta
	formatted_preds = predictions.predictions

	return formatted_preds