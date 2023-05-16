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

	@classmethod
	def from_coco_format(cls, coco_ann, classmap) -> 'Annotation':
		cat_id = coco_ann['category_id']
		classname = classmap[cat_id]
		bbox = coco_ann['bbox'].tolist()
		mask = mask_conversions.rle_to_bin_mask(coco_ann['segmentation'])

		return cls(classname, mask, bbox)