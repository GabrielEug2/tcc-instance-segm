import datetime
import time
from pathlib import Path

import cv2
from tqdm import tqdm

from ..core.managers.pred_manager import PredManager
from .predictors import VALID_MODELS, import_model, MODEL_MAP
from .stats_manager import StatsManager


def run_inference(img_file_or_dir: Path, out_dir: Path, models: list[str] = None):
	"""Runs inference on the requested imgs.

	Args:
		img_file_or_dir (Path): path to an image or dir of images to segment.
		out_dir (Path): directory to save the outputs.
		models (list[str], optional): list of models to use. See
			inference_lib.VALID_MODELS for a list of available models.
			By default, uses all of them.

	Raises:
		ValueError: if an invalid model name was given.
		ImportError: if one of the requested models is not installed properly.
		FileNotFoundError: if no images were found on the provided path.
	"""
	requested_models = VALID_MODELS if models is None else models
	_load_models(requested_models)

	img_files = _get_img_files(img_file_or_dir)

	if not out_dir.exists():
		out_dir.mkdir(parents=True)
	_inference(img_files, out_dir, requested_models)

def _load_models(models):
	for model_name in models:
		try:
			import_model(model_name)
		except ImportError as e:
			raise ImportError(f'"{model_name}" is implemented on the library, but not installed properly.') from e
		except ValueError:
			raise ValueError(f'"{model_name}" is not implemented/setted up correcly. How do you expect to run that.')

def _get_img_files(img_file_or_dir: Path) -> list[Path]:
	if not img_file_or_dir.exists():
		raise FileNotFoundError(f'File or dir not found: "{str(img_file_or_dir)}"')

	if img_file_or_dir.is_dir():
		# If you're gonna run on thousands of images, keeping it a generator
		# would probably be better, but since I only use 200, it's not really
		# relevant
		img_files = list(img_file_or_dir.glob("*.jpg"))
		if len(img_files) == 0:
			raise FileNotFoundError(f'No images found on "{str(img_file_or_dir)}"')
	else:
		img_files = [img_file_or_dir]

	return img_files

def _inference(img_files: list[Path], out_dir: Path, models: list[str]):
	pred_manager = PredManager(out_dir)
	stats_manager = StatsManager()

	n_images = len(img_files)
	stats_manager.set_n_images(n_images)
	print(f'Running {n_images} images on models {models}...')

	for model_name in models:
		print(f'\n\n{model_name}')
		total_time = _run_on_all_imgs(img_files, model_name, pred_manager)
		stats_manager.set_time_for_model(model_name, total_time)

	stats_manager.save(out_dir)

def _run_on_all_imgs(img_files: list[Path], model_name: str, pred_manager: PredManager):
	predictor = MODEL_MAP[model_name]()

	start_time = time.time()
	for img_file in tqdm(img_files):
		img = cv2.imread(str(img_file))
		predictions = predictor.predict(img)

		pred_manager.save(predictions, img_file.stem, model_name)
	total_time = datetime.timedelta(seconds=(time.time() - start_time))

	return total_time