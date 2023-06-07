from abc import ABC, abstractmethod
from pathlib import Path

from segm_lib.core.structures import Annotation, Prediction


class PlotLib(ABC):
	def __init__(self):
		pass

	@abstractmethod
	def plot(self, anns_or_preds: list[Annotation|Prediction], img_file: Path, out_file: Path):
		"""Plots the annotations or predictions on the image.

		Args:
			anns_or_preds (list): list of annotations or predictions to plot.
			img_file (Path): path to the image the annotations will be plotted on,
				in case of a plot on the image.
			out_file (Path): path to the output image. If it doesn't exist, it will
				be created. If it does, it will be overriten.
		"""

	@abstractmethod
	def plot_individual_masks(self, anns_or_preds: list[Annotation|Prediction], img_file: Path, out_dir: Path):
		"""Plots each annotation or prediction separately.

		Args:
			anns_or_preds (list): list of annotations or predictions to plot.
			img_file (Path): path to the image the annotations will be plotted on,
				in case of a plot on the image.
			out_dir (Path): directory where the images will be saved.
		"""