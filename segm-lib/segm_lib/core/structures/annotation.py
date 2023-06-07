

from .object_segmentation import ObjectSegmentation


class Annotation(ObjectSegmentation):
	"""Class representing an annotation in segm_lib format.
	Has the following fields:
		"classname": string,
		"mask": RLE (use core.mask_conversions.rle_to_bin_mask() to convert when needed),
		"bbox": list, [x, y, w, h]
	"""
	def __init__(self, classname: str, mask: dict, bbox: list):
		super().__init__(classname, mask, bbox)
	
	def serializable(self):
		return {
			'classname': self.classname,
			'mask': self.mask,
			'bbox': self.bbox,
		}

	@classmethod
	def from_serializable(cls, seri_ann):
		classname = seri_ann['classname']
		mask = seri_ann['mask']
		bbox = seri_ann['bbox']

		return cls(classname, mask, bbox)

	@classmethod
	def from_coco_format(cls, coco_ann, classname) -> 'Annotation':
		bbox = coco_ann['bbox'] if type(coco_ann['bbox']) == list else coco_ann['bbox'].tolist()
		mask = coco_ann['segmentation']

		return cls(classname, mask, bbox)