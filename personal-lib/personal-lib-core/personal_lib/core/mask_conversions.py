
from pycocotools import mask as coco_mask
import torch

def bin_mask_to_rle(bin_mask: torch.BoolTensor) -> dict:
	rle = coco_mask.encode(bin_mask.numpy().astype('uint8', order='F'))
	rle['counts'] = rle['counts'].decode('ascii')

	return rle

def rle_to_bin_mask(rle: dict) -> torch.BoolTensor:
	rle['counts'] = rle['counts'].encode('utf-8')
	bin_mask = torch.tensor(coco_mask.decode(rle).astype('bool', order='C'))

	return bin_mask

def ann_to_bin_mask(segm, h, w) -> torch.BoolTensor:
	# Código adaptado daqui, mas sem precisar importar as anotações
	# na API do COCO:
	#   https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/coco.py#L413

	if type(segm) == list:
		# polygon -- a single object might consist of multiple parts
		# we merge all parts into one mask rle code
		rles = coco_mask.frPyObjects(segm, h, w)
		rle = coco_mask.merge(rles)
	elif type(segm['counts']) == list:
		# uncompressed RLE
		rle = coco_mask.frPyObjects(segm, h, w)
	else:
		# rle
		rle = segm

	rle['counts'] = rle['counts'].decode('utf-8')

	bin_mask = rle_to_bin_mask(rle)
	
	return bin_mask