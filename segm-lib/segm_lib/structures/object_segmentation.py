import torch

class ObjectSegmentation:
	def __init__(self, classname: str, mask: torch.BoolTensor, bbox: list):
		self.classname = classname
		self.mask = mask
		self.bbox = bbox