from dataclasses import dataclass
from pathlib import Path
import time
import datetime
import json

from tqdm import tqdm
import cv2
from personal_lib import mask_conversions
from personal_lib import plot

from .predictors import MODEL_MAP, load_models

VALID_MODELS = MODEL_MAP.keys()

@dataclass
class InferenceStats():
	n_images: str
	time_for_model: dict[str, datetime.timedelta]

def run_inference(
	img_file_or_dir: Path,
	out_dir: Path,
	models: list[str] = None,
	save_masks: bool = False,
):
	"""Runs inference on the requested imgs.

	Args:
		img_file_or_dir (Path): path to an image or dir of images to segment.
		out_dir (Path): directory to save the outputs.
		models (list[str], optional): list of models to use. See
			inference_lib.VALID_MODELS for a list of available models.
			By default, uses all of them.
		save_masks (bool, optional): whether to save the individual masks too,
		 	as images (True) or not (False). Defaults to False.

	Raises:
		ValueError: if an invalid model name was given.
		ImportError: if one of the requested models is not installed properly.
		FileNotFoundError: if no images were found on the provided path.
	"""
	requested_models = VALID_MODELS if models is None else models
	try:
		load_models(requested_models)
	except (ValueError, ImportError):
		raise

	try:
		img_files = _get_img_files(img_file_or_dir)
	except FileNotFoundError:
		raise

	if not out_dir.exists():
		out_dir.mkdir()

	inference_stats = _inference(img_files, out_dir, requested_models)
	_save_stats(inference_stats, out_dir)
	plot.plot_predictions(out_dir, img_files, save_masks)

def _get_img_files(img_file_or_dir: Path) -> list[Path]:
	if not img_file_or_dir.exists():
		raise FileNotFoundError(f"File or dir not found: \"{str(img_file_or_dir)}\"")

	if img_file_or_dir.is_dir():
		img_files = list(img_file_or_dir.glob("*.jpg"))
		if len(img_files) == 0:
			raise FileNotFoundError(f'No images found on "{str(img_file_or_dir)}"')
	else:
		img_files = [img_file_or_dir]

	return img_files

def _inference(img_files: list[Path], out_dir: Path, models: list[str]):
	n_images = len(img_files)

	print(f"Running on {n_images} images...\n")
	inference_stats = InferenceStats(n_images, {})
	for model_name in models:
		total_time = _run_on_all_imgs(img_files, out_dir, model_name)

		inference_stats.time_for_model[model_name] = total_time

	return inference_stats

def _run_on_all_imgs(img_files: list[Path], out_dir: Path, model_name: str):
	print("\n" + model_name)
	predictor = MODEL_MAP[model_name]()

	start_time = time.time()
	for img_file in tqdm(img_files):
		img = cv2.imread(str(img_file))
		predictions = predictor.predict(img)

		pred_file = out_dir / f"{img_file.stem}_{model_name}.json"
		_save_preds(predictions, pred_file)
	total_time = datetime.timedelta(seconds=(time.time() - start_time))

	return total_time

def _save_preds(predictions, pred_file):
	for pred in predictions:
		pred['mask'] = mask_conversions.bin_mask_to_rle(pred['mask'])

	with pred_file.open('w') as f:
		json.dump(predictions, f)

def _save_stats(inference_stats, out_dir):
	stats_str = _stats_to_str(inference_stats)

	stats_file = out_dir / 'stats.txt'
	with stats_file.open('w') as f:
		f.write(stats_str)

def _stats_to_str(stats: InferenceStats):
	n_images = stats['n_images']

	stats_str = (
		f"{n_images} imagens\n"
		f"{'Modelo'.ljust(10)} {'Tempo total (s)'.ljust(20)} Tempo médio por imagem (s)\n"
	)
	for model_name, total_time in stats['time_for_model'].items():
		average_time = total_time / n_images

		stats_str += f"{model_name.ljust(10)} {str(total_time).ljust(20)} {str(average_time)}\n"

	return stats_str