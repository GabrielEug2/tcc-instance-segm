
from pycocotools import mask as coco_utils

def bin_mask_to_rle(bin_mask):
    # Tensor pra RLE
    rle = coco_utils.encode(bin_mask.numpy().astype('uint8', order='F'))
    rle['counts'] = rle['counts'].decode('ascii')

    return rle