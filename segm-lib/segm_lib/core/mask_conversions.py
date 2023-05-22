import copy

import torch
from pycocotools import mask as coco_mask


def bin_mask_to_rle(bin_mask: torch.BoolTensor) -> dict:
	bin_mask_in_cocoapi_format = bin_mask.numpy().astype('uint8', order='F')
	rle = coco_mask.encode(bin_mask_in_cocoapi_format)
	return _serializable_rle(rle)

def rle_to_bin_mask(rle: dict) -> torch.BoolTensor:
	bin_mask_in_cocoapi_format = coco_mask.decode(_bytes_rle(rle))
	bin_mask = torch.tensor(bin_mask_in_cocoapi_format.astype('bool', order='C'))
	return bin_mask

def ann_to_rle(segm, h, w) -> dict:
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

	return _serializable_rle(rle)

def _serializable_rle(rle):
	# by default, "counts" is in binary, but since I only
	# convert to RLE to *store* the masks, it's better to
	# just return it as a str to automaticaly convert it
	# to JSON later
	serializable_rle = copy.deepcopy(rle)
	serializable_rle['counts'] = rle['counts'].decode('utf-8')
	return serializable_rle

def _bytes_rle(rle):
	serializable_rle = copy.deepcopy(rle)
	serializable_rle['counts'] = rle['counts'].encode('utf-8')
	return serializable_rle
