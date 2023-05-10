
from pathlib import Path
import json
from abc import ABC, abstractmethod

import numpy as np

COCO_CLASSMAP_FILE = Path(__file__).parent / 'coco_classmap_model.json'
with COCO_CLASSMAP_FILE.open('r') as f:
	COCO_CLASSMAP = json.load(f)

# Aqui eu quero pegar o nome dado um ID, então precisa inverter
COCO_CLASSMAP = {str(id_): name for name, id_ in COCO_CLASSMAP.items()}

class Predictor(ABC):
	def __init__(self):
		"""Initializes the model, loading weights and any other
		configurations needed for it's execution."""
		pass
	
	@abstractmethod
	def predict(self, img: np.ndarray) -> list[dict]:
		"""Segments objects on the image.

		Args:
			img (np.ndarray): image in BGR space

		Returns:
			list: list of objects detected on the image, in the format:
				[{
					"classname": string,
					"confidence": float, [0,1]
					"mask": torch.BoolTensor,
					"bbox": [x, y, w, h]
				}]
		"""
		pass
	
	@classmethod
	def cocoid_to_classname(cls, id: int) -> str:
		return COCO_CLASSMAP[str(id)]