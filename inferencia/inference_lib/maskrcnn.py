
import os
import time

import cv2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from inference_lib.format_utils import detectron_to_coco


MASK_RCNN_CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"


def run(img_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MASK_RCNN_CONFIG_FILE))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MASK_RCNN_CONFIG_FILE)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = 'cpu'
    model = DefaultPredictor(cfg)

    img = cv2.imread(img_path)

    start_time = time.time()
    predictions = model(img)
    inference_time = time.time() - start_time

    img_id, _ = os.path.basename(img_path).split('.')
    formatted_output = maskrcnn_to_coco(predictions, img_id)

    return formatted_output, inference_time

def maskrcnn_to_coco(predictions, img_id):
    return detectron_to_coco(predictions, img_id)