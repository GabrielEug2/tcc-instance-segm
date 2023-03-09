
import time
import os
import copy

import cv2
import sys
sys.path.append('./yolact/')
from data import set_cfg
from data import cfg
import torch
from yolact import Yolact
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess

from inference_lib.format_utils import bin_to_rle


YOLACT_CONFIG_FILE = 'yolact_base_config'
YOLACT_WEIGHTS = 'yolact/yolact_base_54_800000.pth'


def run(img_path):
    # Based on:
    # https://github.com/dbolya/yolact/issues/256#issuecomment-567371328
    set_cfg(YOLACT_CONFIG_FILE)
    cfg.mask_proto_debug = False

    model = Yolact()
    model.load_weights(YOLACT_WEIGHTS)
    model.eval()

    img = cv2.imread(img_path)

    start_time = time.time()
    with torch.no_grad():
        frame = torch.from_numpy(img).float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = model(batch)
        # Essas predições ainda não são finais, falta converter
        # as máscaras pro formato certo.
    h, w, _ = img.shape
    predictions = postprocess(preds, w, h, score_threshold = 0.5)
    inference_time = time.time() - start_time

    img_id, _ = os.path.basename(img_path).split('.')
    formatted_output = yolact_to_coco(predictions, img_id)

    return formatted_output, inference_time

def yolact_to_coco(predictions, img_id):   
    # Model output - https://github.com/dbolya/yolact/blob/master/layers/output_utils.py
    # COCO format - https://cocodataset.org/#format-results
    coco_style_predictions = []
    temp_coco_prediction = {}

    for i in range(len(predictions[0])):
        pred_class = predictions[0][i].item()
        score = predictions[1][i].item()
        bin_mask = predictions[3][i]

        temp_coco_prediction['image_id'] = img_id
        temp_coco_prediction['category_id'] = pred_class
        temp_coco_prediction['segmentation'] = bin_to_rle(bin_mask)
        temp_coco_prediction['score'] = score

        coco_style_predictions.append(copy.copy(temp_coco_prediction))

    return coco_style_predictions