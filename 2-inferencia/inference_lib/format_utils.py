
from pycocotools import mask as rle_utils

def bin_mask_to_rle(bin_mask):
    rle = rle_utils.encode(bin_mask.numpy().astype('uint8', order='F'))
    rle['counts'] = rle['counts'].decode('ascii')

    return rle