
import copy

from pycocotools import mask


def detectron_to_coco(predictions, img_id):
    # Model output - https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-format
    # COCO format - https://cocodataset.org/#format-results

    coco_style_predictions = []
    temp_coco_prediction = {}

    for i in range(len(predictions['instances'])):
        pred_class = predictions['instances'].pred_classes[i].item()
        score = predictions['instances'].scores[i].item()
        bin_mask = predictions['instances'].pred_masks[i]

        temp_coco_prediction['image_id'] = img_id
        temp_coco_prediction['category_id'] = pred_class
        temp_coco_prediction['segmentation'] = bin_to_rle(bin_mask)
        temp_coco_prediction['score'] = score

        coco_style_predictions.append(copy.copy(temp_coco_prediction))

    return coco_style_predictions

def bin_to_rle(bin_mask):
    rle_mask = mask.encode(bin_mask.numpy().astype('uint8', order='F'))
    rle_mask['counts'] = rle_mask['counts'].decode('ascii')

    return rle_mask