
from pycocotools import mask as rle_utils

def rle_to_bin_mask(rle):
    rle['counts'] = rle['counts'].encode('ascii')
    bin_mask = rle_utils.decode(rle)

    return bin_mask