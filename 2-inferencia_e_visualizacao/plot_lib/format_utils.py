
from pycocotools import mask as mask_utils

def rle_to_bin_mask(rle):
	# Compact RLE to bin tensor
	rle['counts'] = rle['counts'].encode('utf-8')
	bin_mask = mask_utils.decode(rle).astype('bool', order='C')

	return bin_mask

def ann_to_bin_mask(segm, img_desc):
	# Código adaptado daqui, mas sem precisar importar as anotações
	# na API do COCO:
	#   https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/coco.py#L413

	h, w = img_desc['height'], img_desc['width']

	if type(segm) == list:
		# polygon -- a single object might consist of multiple parts
		# we merge all parts into one mask rle code
		rles = mask_utils.frPyObjects(segm, h, w)
		rle = mask_utils.merge(rles)
	elif type(segm['counts']) == list:
		# uncompressed RLE
		rle = mask_utils.frPyObjects(segm, h, w)
	else:
		# rle
		rle = segm

	rle['counts'] = rle['counts'].decode('utf-8')

	bin_mask = rle_to_bin_mask(rle)
	
	return bin_mask