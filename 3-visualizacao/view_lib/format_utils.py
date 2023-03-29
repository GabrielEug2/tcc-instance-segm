
from pycocotools import mask as coco_utils
from pycocotools.coco import COCO

def rle_to_bin_mask(rle):
    rle['counts'] = rle['counts'].encode('ascii')
    bin_mask = coco_utils.decode(rle).astype('bool', order='C')

    return bin_mask

def polygon_to_rle(polygon, height, width):
    # See:
    #   https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/coco.py#L413

    # TODO Debug
    coco = COCO(ann_file)
    coco.annToRle(ann)
    rle = coco_utils.frPyObjects(polygon, height, width)
    rle['counts'] = rle['counts'].decode('ascii')

    return rle

def bin_mask_to_rle(bin_mask):
    rle = coco_utils.encode(bin_mask)
    rle['counts'] = rle['counts'].decode('ascii')

    return rle