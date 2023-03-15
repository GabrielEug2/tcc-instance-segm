
import time

import cv2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from .config import config
from . import format_utils

def predict(img_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config['maskrcnn']['config_file']))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config['maskrcnn']['config_file'])
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = 'cpu'
    model = DefaultPredictor(cfg)

    img = cv2.imread(str(img_path))

    start_time = time.time()
    predictions = model(img)
    inference_time = time.time() - start_time

    img_id = img_path.stem
    predictions = format_output(predictions, img_id)

    return predictions, inference_time

def format_output(predictions, img_id):    
    return format_utils.detectron_to_coco(predictions, img_id)