
import torch

class Prediction():
	"""Class representing a prediction, ONLY for inference purposes.
	Everywhere else, if should be manipulated as a
	segm_lib.core.structures.Prediction.

	Has the following fields:
		"classname": string,
		"confidence": float, [0,1]
		"mask": torch.BoolTensor,
		"bbox": list, [x, y, w, h]
	"""
	def __init__(self, classname: str, confidence: float, mask: torch.BoolTensor, bbox: list):
		self.classname = classname
		self.confidence = confidence
		self.mask = mask
		self.bbox = bbox