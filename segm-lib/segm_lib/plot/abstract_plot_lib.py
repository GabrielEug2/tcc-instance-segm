from abc import ABC, abstractmethod
from pathlib import Path

from segm_lib.structures import Annotation, Prediction

class AbstractPlotLib(ABC):
	@abstractmethod
	def plot(
		anns_or_preds: list[Annotation]|list[Prediction],
		img_file: Path,
		out_file: Path
	):
		"""Plots the annotations or predictions on the image.

		Args:
			anns_or_preds (list): list of annotations or predictions to plot.
			img_file (Path): path to the image the annotations will be plotted on.
			out_file (Path): path to the output image.
		"""

	@abstractmethod
	def plot_individual_masks(
		anns_or_preds: list[Annotation]|list[Prediction],
		out_dir: Path,
		img_file: Path
	):
		"""Plots each annotation or prediction separately.

		Args:
			anns_or_preds (list): list of annotations or predictions to plot.
			out_dir (Path): directory where the images will be saved
			out_file (Path): path to the image the annotations will be plotted on,
				in case of a plot on the image (not just a bin image).
		"""