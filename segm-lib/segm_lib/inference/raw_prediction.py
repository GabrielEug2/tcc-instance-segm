
import torch

class RawPrediction():
	"""Class representing a prediction, ONLY for inference purposes. After inference, they're
	saved in a more compact format, and should be manipulated with the appropriated class
	(segm_lib.core.structures.prediction.Prediction).

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