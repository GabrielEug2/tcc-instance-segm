import torch

from .. import mask_conversions
from .object_segmentation import ObjectSegmentation


class Prediction(ObjectSegmentation):
	"""Class representing a prediction in segm_lib format.
	Has the following fields:
		"classname": string,
		"confidence": float, [0,1]
		"mask": torch.BoolTensor,
		"bbox": list, [x, y, w, h]
	"""
	def __init__(self, classname: str, confidence: float, mask: torch.BoolTensor, bbox: list):
		super().__init__(classname, mask, bbox)
		self.confidence = confidence

	def serializable(self):
		return {
			'classname': self.classname,
			'confidence': self.confidence,
			'mask': mask_conversions.bin_mask_to_rle(self.mask),
			'bbox': self.bbox,
		}
	
	@classmethod
	def from_coco_format(cls, coco_pred, classmap) -> 'Prediction':
		classname = classmap[coco_pred['category_id']]
		confidence = coco_pred['score']
		bbox = coco_pred['bbox'] if type(coco_pred['bbox']) == list else coco_pred['bbox'].tolist()
		mask = mask_conversions.rle_to_bin_mask(coco_pred['segmentation'])

		return cls(classname, confidence, mask, bbox)