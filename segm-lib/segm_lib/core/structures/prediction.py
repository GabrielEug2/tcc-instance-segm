

from .object_segmentation import ObjectSegmentation


class Prediction(ObjectSegmentation):
	"""Class representing a prediction in segm_lib format.
	Has the following fields:
		"classname": string,
		"confidence": float, [0,1]
		"mask": RLE (use core.mask_conversions.rle_to_bin_mask() to convert when needed),
		"bbox": list, [x, y, w, h]
	"""
	def __init__(self, classname: str, confidence: float, mask: dict, bbox: list):
		super().__init__(classname, mask, bbox)
		self.confidence = confidence

	def serializable(self):
		return {
			'classname': self.classname,
			'confidence': self.confidence,
			'mask': self.mask,
			'bbox': self.bbox,
		}
	
	@classmethod
	def from_serializable(cls, seri_pred):
		classname = seri_pred['classname']
		confidence = seri_pred['confidence']
		mask = seri_pred['mask']
		bbox = seri_pred['bbox']

		return cls(classname, confidence, mask, bbox)
	
	@classmethod
	def from_coco_format(cls, coco_pred, classname) -> 'Prediction':
		confidence = coco_pred['score']
		bbox = coco_pred['bbox'] if type(coco_pred['bbox']) == list else coco_pred['bbox'].tolist()
		mask = coco_pred['segmentation']

		return cls(classname, confidence, mask, bbox)