from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm

from segm_lib.structures import Annotation, Prediction
from segm_lib.ann_manager import AnnManager
from segm_lib.pred_manager import PredManager

class AbstractPlotLib(ABC):
	def __init__(self):
		pass

	def plot_annotations(self, ann_dir: Path, img_dir: Path, out_dir: Path):
		ann_manager = AnnManager(ann_dir)
		img_files = img_dir.glob('*.jpg')
		n_images = sum(1 for _ in img_dir.glob('*.jpg')) # need to glob again cause img_files is a generator, I don't want to consume it

		for img_file in tqdm(img_files, total=n_images):
			annotations = ann_manager.load(img_file.stem)
			if len(annotations) == 0:
				print(f"No annotations found on \"{str(ann_dir)} for image \"{str(img_file)}\". Skipping")
				continue

			annotated_img_file = out_dir / img_file.stem / "groundtruth.jpg"
			self._plot(annotations, img_file, annotated_img_file)

			mask_out_dir = out_dir / img_file.stem / "groundtruth_masks"
			mask_out_dir.mkdir(parents=True, exist_ok=True)
			self._plot_individual_masks(annotations, mask_out_dir, img_file)

	def plot_predictions(self, pred_dir: Path, img_dir: Path, out_dir: Path):
		pred_manager = PredManager(pred_dir)
		n_images = sum(1 for _ in img_dir.glob('*.jpg'))

		model_names = pred_manager.get_model_names()
		if len(model_names) == 0:
			print(f"No predictions found on {pred_dir}.")
			return
		print(f"Found predictions for models {model_names}")

		for model_name in model_names:
			print(f"\nPlotting predictions from model {model_name}...")
			img_files = img_dir.glob('*.jpg') # needs to glob each time because it's a generator

			for img_file in tqdm(img_files, total=n_images):
				predictions = pred_manager.load(img_file.stem, model_name)
				if len(predictions) == 0:
					print(f"No predictions found on \"{str(pred_dir)}\" for image \"{str(img_file)}\". Skipping")
					continue

				predictions_img_file = out_dir / img_file.stem / f"{model_name}.jpg"
				self._plot(predictions, img_file, predictions_img_file)

				mask_out_dir = out_dir / img_file.stem / f"{model_name}_masks"
				self._plot_individual_masks(predictions, mask_out_dir, img_file)

	@abstractmethod
	def _plot(anns_or_preds: list[Annotation]|list[Prediction], img_file: Path, out_file: Path):
		"""Plots the annotations or predictions on the image.

		Args:
			anns_or_preds (list): list of annotations or predictions to plot.
			img_file (Path): path to the image the annotations will be plotted on.
			out_file (Path): path to the output image. If it doesn't exist, it will
				be created. If it does, it will be overriten.
		"""

	@abstractmethod
	def _plot_individual_masks(anns_or_preds: list[Annotation]|list[Prediction], out_dir: Path, img_file: Path):
		"""Plots each annotation or prediction separately.

		Args:
			anns_or_preds (list): list of annotations or predictions to plot.
			out_dir (Path): directory where the images will be saved
			img_file (Path): path to the image the annotations will be plotted on,
				in case of a plot on the image. If it doesn't exist, it will be
				created. If it does, it will be overriten.
		"""