
from pycocotools import mask as rle_utils


def bin_mask_to_rle(bin_mask):
    rle_mask = rle_utils.encode(bin_mask.numpy().astype('uint8', order='F'))
    rle_mask['counts'] = rle_mask['counts'].decode('ascii')

    return rle_mask

def rle_to_bin_mask(rle):
    rle['counts'] = rle['counts'].encode('ascii')
    bin_mask = rle_utils.decode(rle)

    return bin_mask