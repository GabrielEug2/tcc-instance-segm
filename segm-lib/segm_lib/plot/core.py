from pathlib import Path

from tqdm import tqdm

from segm_lib.annotations import Annotations
from segm_lib.predictions import Predictions
from .detectron_plot_lib import DetectronPlotLib

plot_lib = DetectronPlotLib()

def plot_annotations(ann_dir: Path, img_dir: Path, out_dir: Path):
	ann_manager = Annotations(ann_dir)
	img_files = img_dir.glob('*.jpg')
	n_images = len(list(img_dir.glob('*.jpg'))) # need to glob again cause img_files is a generator, I don't want to consume it

	for img_file in tqdm(img_files, total=n_images):
		anns = ann_manager.load(img_file.stem)
		if len(anns) == 0:
			print(f"No annotations found on \"{str(ann_dir)} for image \"{str(img_file)}\". Skipping")
			continue

		formatted_anns = _ann_to_plot_format(anns)
		
		annotated_img_file = out_dir / img_file.stem / "groundtruth.jpg"
		annotated_img_file.parent.mkdir(parents=True, exist_ok=True)
		plot_lib.plot(formatted_anns, img_file, annotated_img_file)

		mask_out_dir = out_dir / img_file.stem / "groundtruth_masks"
		mask_out_dir.mkdir(parents=True, exist_ok=True)
		plot_lib.plot_individual_masks(formatted_anns, mask_out_dir, img_file)

def _ann_to_plot_format(anns):
	formatted_anns = []
	for ann in anns:
		classname = ann['classname']
		confidence = 100.0
		bbox = ann['bbox']
		mask = ann['mask']

		formatted_anns.append({
			'classname': classname,
			'confidence': confidence,
			'mask': mask,
			'bbox': bbox,
		})

	return formatted_anns

def plot_predictions(pred_dir: Path, img_dir: Path, out_dir: Path):
	pred_manager = Predictions(pred_dir)
	n_images = len(list(img_dir.glob('*.jpg')))
	model_names = pred_manager.get_model_names()
	print(f"Found predictions for models {model_names}")

	for model_name in model_names:
		print(f"\nPlotting predictions from model {model_name}...")
		img_files = img_dir.glob('*.jpg') # needs to glob each time because it's a generator

		for img_file in tqdm(img_files, total=n_images):
			predictions = pred_manager.load(img_file.stem, model_name)
			if len(predictions) == 0:
				print(f"No predictions found on \"{str(pred_dir)}\" for image \"{str(img_file)}\". Skipping")
				continue

			# já está no formato que eu quero

			predictions_img_file = out_dir / img_file.stem / f"{model_name}.jpg"
			predictions_img_file.parent.mkdir(parents=True, exist_ok=True)
			plot_lib.plot(predictions, img_file, predictions_img_file)

			mask_out_dir = out_dir / img_file.stem / f"{model_name}_masks"
			mask_out_dir.mkdir(parents=True, exist_ok=True)
			plot_lib.plot_individual_masks(predictions, mask_out_dir, img_file)