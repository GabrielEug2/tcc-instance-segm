
from pycocotools import mask as coco_utils

def rle_to_bin_mask(rle):
    # RLE pra tensor
    rle['counts'] = rle['counts'].encode('ascii')
    bin_mask = coco_utils.decode(rle).astype('bool', order='C')

    return bin_mask

def polygon_to_rle(polygon, height, width):
    # See:
    #   https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/coco.py#L413

    # TODO Debug
    rle = coco_utils.frPyObjects(polygon, height, width)
    rle['counts'] = rle['counts'].decode('ascii')

    return rle