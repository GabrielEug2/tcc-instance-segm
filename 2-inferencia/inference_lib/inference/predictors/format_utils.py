
from pycocotools import mask as mask_utils

def bin_mask_to_rle(bin_mask):
    # Bin tensor to COCO compact RLE
    rle = mask_utils.encode(bin_mask.numpy().astype('uint8', order='F'))
    rle['counts'] = rle['counts'].decode('ascii')

    return rle