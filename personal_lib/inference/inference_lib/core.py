from pathlib import Path
import time
import datetime
import json

from tqdm import tqdm
import cv2
import plot_lib

from .predictors import MODEL_MAP, load_models

VALID_MODELS = MODEL_MAP.keys()

def run_inference(
	img_file_or_dir: Path,
	out_dir: Path,
	models: list[str] = None,
	compressed_masks: bool = False,
	save_masks: bool = False,
):
	"""Runs inference on the requested imgs.

	Args:
		img_file_or_dir (Path): path to an image or dir of images to segment.
		out_dir (Path): directory to save the outputs.
		models (list[str], optional): list of models to use. See
			inference_lib.VALID_MODELS for a list of available models.
			By default, uses all of them.
		compressed_masks (bool, optional): whether to save the masks in
			COCO RLE format, for smaller file size (True) or in raw, binary
			form (False). Defaults to False.
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

	if compressed_masks:
		try:
			import conversions
		except ImportError:
			print("Can't compress masks without conversions lib. Using regular masks instead.")
			compressed_masks = False

	img_file_or_dir = Path(img_file_or_dir)
	try:
		img_files = _get_img_files(img_file_or_dir)
	except FileNotFoundError:
		raise

	out_dir = Path(out_dir)
	if not out_dir.exists():
		out_dir.mkdir()

	_inference(img_files, out_dir, requested_models)
	_post_processing(img_files, out_dir, save_masks, compressed_masks)


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
	inference_stats = {
		'n_images': n_images,
		'model_data': []
	}
	for model_name in models:
		total_time = _run_on_all_imgs(img_files, out_dir, model_name)

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

def _run_on_all_imgs(img_files: list[Path], out_dir: Path, model_name: str):
	print("\n" + model_name)
	predictor = MODEL_MAP[model_name]()

	start_time = time.time()
	for img_file in tqdm(img_files):
		img = cv2.imread(str(img_file))
		predictions = predictor.predict(img)

		pred_file = out_dir / f"{img_file.stem}_{model_name}.json"
		with pred_file.open('w') as f:
			json.dump(predictions, f)
	total_time = datetime.timedelta(seconds=(time.time() - start_time))

	return total_time

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

def _post_processing(img_files: list[Path], out_dir: Path, save_masks: bool, compressed_masks: bool):
	for img_file in img_files:
		pred_files = out_dir.glob(f"{img_file.stem}_*.json")

		for pred_file in pred_files:
			model_name = pred_file.stem.split('_')[1]
			with pred_file.open('r') as f:
				predictions = json.load(f)

			predictions_img = plot_lib.plot(img_file, predictions)
			predictions_img_file = out_dir / f"{img_file.stem}_{model_name}.jpg"
			cv2.imwrite(str(predictions_img_file), predictions_img)

			if save_masks:
				mask_out_dir = out_dir / f"{img_file.stem}_{model_name}_masks"
				mask_out_dir.mkdir()
				_save_masks(predictions, mask_out_dir)
			if compressed_masks:
				_compress_masks(predictions, pred_file)

def _save_masks(predictions, out_dir):
	count_per_class = {}

	for pred in predictions:
		classname = pred['classname']
		mask = pred['mask']
		
		count_per_class[classname] += 1
		mask_file = out_dir / f"{classname}_{count_per_class[classname]}.jpg"
		cv2.imwrite(str(mask_file), mask)

def _compress_masks(predictions, pred_file):
	for pred in predictions:
		pred['mask'] = conversions.bin_mask_to_rle(pred['mask'])

	with pred_file.open('w') as f:
		json.dump(predictions, f)