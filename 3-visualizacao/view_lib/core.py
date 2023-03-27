
import json
from pathlib import Path

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes
import torch
import numpy as np

from .format_utils import rle_to_bin_mask

def plot_annotations(img_path, ann_path, out_dir):
    img_file = Path(img_path)
    ann_file = Path(ann_path)
    out_dir = Path(out_dir)

    with ann_file.open('r') as f:
        anns = json.load(f)

    anns['images']

    # search in images by img.stem to get id
    # search in annotations by id to find annotations
    # instances object
    # plot

    pass

def plot_predictions():
    # for pred_file
    # output_dir / f"{img_path.stem}_*_pred.json"
    #   load predictions
    #   with predictions_file.open('w') as f:
    #     json.dump(predictions, f)


    # predictions_img = _plot(predictions, img_path)
    # predictions_img_file = output_dir / f"{img_path.stem}_{model['name']}_pred.jpg"
    # cv2.imwrite(str(predictions_img_file), predictions_img)
    pass

def _plot_img():
    pass
            
def _save_masks():
    #     i = 1
    #     for prediction in predictions:
    #         mask = rle_to_bin_mask(prediction['mask'])
    #         mask_filename = output_dir / f"{img_path.stem}_{model['name']}_masks" / f"{i}.jpg"
    #         cv2.imwrite(mask_filename, mask)
    pass
            

def _plot(predictions, img_path):
    classes = []
    scores = []
    masks = []
    boxes = []
    for prediction in predictions:
        classes.append(prediction['class_id'] - 1) # Para plotar, precisa estar em [0,N)
        scores.append(prediction['confidence'])
        masks.append(rle_to_bin_mask(prediction['mask']))
        boxes.append(prediction['bbox'])

    img = cv2.imread(str(img_path))
    h, w, _ = img.shape
    instances = Instances((h, w))
    instances.pred_classes = torch.tensor(classes, dtype=torch.int)
    instances.scores = torch.tensor(scores, dtype=torch.float)
    instances.pred_masks = torch.tensor(np.array(masks), dtype=torch.bool)
    instances.pred_boxes = Boxes(torch.tensor(boxes, dtype=torch.float))

    v = Visualizer(img, MetadataCatalog.get('coco_2017_test'))
    vis_out = v.draw_instance_predictions(instances)
    predictions_img = vis_out.get_image()

    return predictions_img










# def load_predictions(dataset, predictions_dir, dataset_dir):
#     raw_pred_files = []
#     for file in Path(predictions_dir).glob('*.json'):
#         if not file.name.endswith(FIXED_ANN_SUFIX):
#             raw_pred_files.append(file)

#     if len(raw_pred_files) == 0:
#         print(f"No predictions found on \"{predictions_dir}\".")
#         exit()

#     fixed_pred_files = []
#     for raw_pred_file in raw_pred_files:
#         fixed_pred_files.append(Path(
#             predictions_dir,
#             raw_pred_file.name.removesuffix(NORMAL_PRED_SUFFIX) + FIXED_PRED_SUFIX
#         ))

#     if any(not f.exists() for f in fixed_pred_files):
#         fixed_ann_file = Path(dataset_dir, REL_ANN_PATH + FIXED_ANN_SUFIX)
#         conversions.fix_predictions(raw_pred_files, fixed_ann_file)

#     classes = dataset.default_classes
#     id_detections_map = {}
#     for pred_file in fixed_pred_files:
#         with pred_file.open('r') as f:
#             predictions = json.load(f)

#         model_name = pred_file.name.removesuffix(FIXED_PRED_SUFIX)
#         for prediction in predictions:
#             det_list = id_detections_map.get(prediction['image_id'], [])
#             det_list.append(
#                 fo.Detection(
#                     mask=conversions.rle_to_bin(prediction['segmentation']),
#                     label=classes[prediction['category_id']],
#                     confidence=prediction['score']
#                 )
#             )
#             id_detections_map[prediction['image_id']] = det_list

#         for sample in dataset:
#             # sample = dataset.match(F("ground_truth_coco_id") == prediction['image_id']).first()
#             coco_id = sample.ground_truth_coco_id
#             sample[model_name] = fo.Detections(detections=id_detections_map[coco_id])
#             sample.save()

#         # fo_coco_utils.add_coco_labels(
#         #     dataset,
#         #     model_name,
#         #     predictions,
#         #     dataset.default_classes,
#         #     label_type='segmentations',
#         #     coco_id_field='ground_truth_coco_id'
#         # )

#     return dataset