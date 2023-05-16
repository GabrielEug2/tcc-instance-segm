import numpy as np
import torch

from segm_lib import mask_conversions

from .object_segmentation import ObjectSegmentation

class Annotation(ObjectSegmentation):
	"""Class representing an annotation in segm_lib format.
	Has the following fields:
		"classname": string,
		"mask": torch.BoolTensor,
		"bbox": list, [x, y, w, h]
	"""
	def __init__(self, classname: str, mask: torch.BoolTensor, bbox: list):
		super().__init__(classname, mask, bbox)
	
	def serializable(self):
		return {
			'classname': self.classname,
			'mask': mask_conversions.bin_mask_to_rle(self.mask),
			'bbox': self.bbox,
		}
	
	@classmethod
	def from_coco_format(cls, coco_ann, classmap) -> 'Annotation':
		classname = classmap[coco_ann['category_id']]
		bbox = coco_ann['bbox'] if type(coco_ann['bbox']) == list else coco_ann['bbox'].tolist()
		mask = mask_conversions.rle_to_bin_mask(coco_ann['segmentation'])

		return cls(classname, mask, bbox)