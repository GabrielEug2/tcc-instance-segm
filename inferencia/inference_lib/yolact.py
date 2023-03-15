from pathlib import Path
import time
import copy

import cv2
import torch

from .config import config
from .format_utils import bin_mask_to_rle

import sys
sys.path.insert(0, config['yolact']['dir'])
from data import set_cfg
from data import cfg
from yolact import Yolact
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess


def predict(img_path):
    # Based on:
    # https://github.com/dbolya/yolact/issues/256#issuecomment-567371328
    set_cfg(config['yolact']['config_name'])
    cfg.mask_proto_debug = False

    model = Yolact()
    model.load_weights(str(Path(config['yolact']['dir'], config['yolact']['weights_file'])))
    model.eval()

    img = cv2.imread(str(img_path))

    start_time = time.time()
    with torch.no_grad():
        frame = torch.from_numpy(img).float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = model(batch)
        # Essas predições ainda não são finais, falta converter
        # as máscaras pro formato certo.
    h, w, _ = img.shape
    classes, scores, _, masks = postprocess(preds, w, h, score_threshold = 0.5)
    predictions = {'classes': classes, 'scores': scores, 'masks': masks}
    inference_time = time.time() - start_time

    img_id = img_path.stem
    predictions = format_output(predictions, img_id)

    return predictions, inference_time

def format_output(predictions, img_id):
    # Model output - https://github.com/dbolya/yolact/blob/master/layers/output_utils.py
    # COCO format - https://cocodataset.org/#format-results
    coco_style_predictions = []
    temp_coco_prediction = {}

    for i in range(len(predictions['masks'])):
        pred_class = predictions['classes'][i].item()
        score = predictions['scores'][i].item()
        bin_mask = predictions['masks'][i]

        temp_coco_prediction['image_id'] = img_id
        temp_coco_prediction['category_id'] = pred_class
        temp_coco_prediction['segmentation'] = bin_mask_to_rle(bin_mask)
        temp_coco_prediction['score'] = score

        coco_style_predictions.append(copy.copy(temp_coco_prediction))

    return coco_style_predictions