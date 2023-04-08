from pathlib import Path
import time
import datetime
import json

from tqdm import tqdm
import cv2
import plot_lib

from .predictors import MODEL_MAP

VALID_MODELS = MODEL_MAP.keys()

def run_inference(img_file_or_dir: str, out_dir: str, models: list[str] = None):
	"""Runs inference on the requested imgs.

	Args:
		img_file_or_dir (str): path to image or dir of images to segment.
		out_dir (str): directory to save the outputs.
		models (list[str], optional): list of models to use. See
			inference_lib.VALID_MODELS for a list of available models.
			By default, uses all of them.

	Raises:
		FileNotFoundError: if no images were found on the provided path.
		ValueError: if an invalid model name was given.
	"""
	img_file_or_dir = Path(img_file_or_dir)
	out_dir = Path(out_dir)
	requested_models = VALID_MODELS if models is None else models

	if any(requested_models not in VALID_MODELS):
		raise ValueError(f'Invalid model list: "{requested_models}". Must be a subset of {VALID_MODELS}')

	img_files = _get_img_files(img_file_or_dir)

	if not out_dir.exists():
		out_dir.mkdir()

	_actual_inference(img_files, out_dir, requested_models)
	plot_lib.predictions.plot(img_files, pred_dir=out_dir, out_dir=out_dir)

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

def _actual_inference(img_files: list[Path], out_dir: Path, models: list[str]) -> list[Path]:
	n_images = len(img_files)

	print(f"Running on {n_images} images...\n")
	inference_stats = {
		'n_images': n_images,
		'model_data': []
	}
	pred_files = []
	for model_name in models:
		print("\n" + model_name)
		predictor = MODEL_MAP[model_name]

		start_time = time.time()
		for img_file in tqdm(img_files):
			img = cv2.imread(str(img_file))
			predictions = predictor.predict(img)

			pred_file = out_dir / f"{img_file.stem}_{model_name}.json"
			with pred_file.open('w') as f:
				json.dump(predictions, f)
			pred_files.append(pred_file)

		total_time = datetime.timedelta(seconds=(time.time() - start_time))

		average_time = total_time / n_images
		inference_stats['model_data'].append({
			'name': model_name,
			'total': total_time,
			'average': average_time,
		})

	stats_str = _stats_to_str(inference_stats)    
	stats_file = out_dir / 'stats.txt'
	with stats_file.open('w') as f:
		f.write(stats_str)

	return pred_files

def _stats_to_str(stats):
	stats_str = (
		f"{stats['n_images']} imagens\n"
		f"{'Modelo'.ljust(10)} {'Tempo total (s)'.ljust(20)} Tempo médio por imagem (s)\n"
	)
	for model in stats['model_data']:
		name = model['name']
		total = model['total']
		average = model['average']
		stats_str += f"{name.ljust(10)} {str(total).ljust(20)} {str(average)}\n"