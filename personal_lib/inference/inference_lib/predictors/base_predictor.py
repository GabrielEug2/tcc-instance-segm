
from pathlib import Path
import json
from abc import ABC, abstractmethod

COCO_CLASSMAP_FILE = Path(__file__).parent / 'coco_classmap_model.json'
with COCO_CLASSMAP_FILE.open('r') as f:
	COCO_CLASSMAP = json.load(f)

class Predictor(ABC):
	def __init__(self):
		"""Initializes the model, loading weights and any other
		configurations needed for it's execution."""
		pass
	
	@abstractmethod
	def predict(self, img):
		"""Segments objects on the image.

		Args:
			img (np.ndarray): image in BGR space

		Returns:
			list: list of objects detected on the image, in the format:
				{"classname": string,
				"confidence": float, between 0 and 1,
				"mask": torch.BoolTensor,
				"bbox": [x1, y1, x2, y2]}
		"""
		pass
	
	@classmethod
	def cocoid_to_classname(cls, id):
		return COCO_CLASSMAP[str(id)]