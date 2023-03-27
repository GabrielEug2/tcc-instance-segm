
import json
from pathlib import Path

import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes
import torch
import numpy as np

from .format_utils import polygon_to_rle, rle_to_bin_mask

def plot_annotations(img_file_str, ann_file_str, out_dir_str):
    img_file = Path(img_file_str)
    ann_file = Path(ann_file_str)
    out_dir = Path(out_dir_str)

    with ann_file.open('r') as f:
        anns = json.load(f)

    requested_img_id = None
    for img_desc in anns['images']:
        if img_desc['filename'] == img_file.stem:
            requested_img_id = img_desc['id']
            break
    if requested_img_id == None:
        print("Imagem não encontrada no arquivo de annotations")
        exit(1)
    
    relevant_anns = []
    for ann in anns['annotations']:
        if ann['image_id'] == requested_img_id:
            relevant_anns.append(ann)
    if len(relevant_anns) == 0:
        print("A imagem não possui annotations no arquivo informado")
        exit(2)

    # Converte as annotations pro formato de prediction pra plotar da mesma forma
    anns_in_pred_format = []
    img = cv2.imread(str(img_file))
    h, w, _ = img.shape
    for ann in relevant_anns:
        class_id = ann['category_id'] - 1 # Para plotar, precisa estar em [0,N)
        confidence = ann['score']
        mask = polygon_to_rle(ann['segmentation'], h, w)
        bbox = ann['bbox']

        pred = {
            'class_id': class_id,
            'confidence': confidence,
            'mask': mask,
            'bbox': bbox,
        }
        anns_in_pred_format.append(pred)

    annotated_img = _plot(anns_in_pred_format, img_file)
    annotated_img_file = out_dir  / f"{img_file.stem}_groundtruth.jpg"
    cv2.imwrite(str(annotated_img_file), annotated_img)

    return

def plot_predictions(img_file_str, pred_dir_str, out_dir_str):
    img_file = Path(img_file_str)
    pred_dir = Path(pred_dir_str)
    out_dir = Path(out_dir_str)

    pred_files = list(pred_dir.glob(f"{img_file.stem}_*_pred.json"))
    if len(pred_files) == 0:
        print(f"No predictions found on \"{pred_dir_str}\".")
        exit()

    for pred_file in pred_files:
        with pred_file.open('r') as f:
            predictions = json.load(f)

        predictions_img = _plot(predictions, img_file)
        model_name = img_file.stem.split('_')[1]
        predictions_img_file = out_dir / f"{img_file.stem}_{model_name}_pred.jpg"
        cv2.imwrite(str(predictions_img_file), predictions_img)
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