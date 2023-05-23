
import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ...core.structures import Prediction

COCO_CLASSMAP_FILE = Path(__file__).parent / 'coco_classmap_normalized.json'
with COCO_CLASSMAP_FILE.open('r') as f:
	COCO_CLASSMAP = json.load(f)
# Aqui eu quero pegar o nome dado um ID, então precisa inverter
COCO_CLASSMAP = {str(id_): name for name, id_ in COCO_CLASSMAP.items()}

class Predictor(ABC):
	def __init__(self):
		"""Initializes the model, loading weights and any other
		configurations needed for it's execution."""
	
	@abstractmethod
	def predict(self, img: np.ndarray) -> list[Prediction]:
		"""Segments objects on the image.

		Args:
			img (np.ndarray): image in BGR space.

		Returns:
			list[Prediction]: list of objects detected on the image.
		"""
	
	@classmethod
	def cocoid_to_classname(cls, id: int) -> str:
		return COCO_CLASSMAP[str(id)]