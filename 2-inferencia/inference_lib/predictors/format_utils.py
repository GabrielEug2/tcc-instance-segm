
from pycocotools import mask

# def detectron_to_coco(predictions, img_id):
#     # Model output - https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-format
#     # COCO format - https://cocodataset.org/#format-results
#     coco_style_predictions = []

#     for i in range(len(predictions['instances'])):
#         pred_class = predictions['instances'].pred_classes[i].item()
#         score = predictions['instances'].scores[i].item()
#         bin_mask = predictions['instances'].pred_masks[i]

#         coco_style_prediction = to_coco_format(img_id, pred_class, score, bin_mask)

#         coco_style_predictions.append(coco_style_prediction)

#     return coco_style_predictions

# def to_coco_format(img_id, pred_class, score, bin_mask):
#     coco_prediction = {}
#     coco_prediction['image_id'] = img_id
#     coco_prediction['category_id'] = pred_class
#     coco_prediction['segmentation'] = bin_mask_to_rle(bin_mask)
#     coco_prediction['score'] = score

#     return coco_prediction

def plot(predictions):
    pass