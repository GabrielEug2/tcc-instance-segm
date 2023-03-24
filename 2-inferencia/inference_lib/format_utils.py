
from pycocotools import mask as rle_utils

def bin_mask_to_rle(bin_mask):
    # TODO: inspect mask
    rle = rle_utils.encode(bin_mask.numpy().astype('uint8', order='F'))
    rle['counts'] = rle['counts'].decode('ascii')

    return rle

def rle_to_bin_mask(rle):
    # TODO: inspect mask
    # Melhor representação binária seria 0/1 ao invés de true/talse
    rle['counts'] = rle['counts'].encode('ascii')
    bin_mask = rle_utils.decode(rle)

    return bin_mask