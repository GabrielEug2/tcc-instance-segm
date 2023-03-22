
import time

import cv2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from .predictor import Predictor
from .config import config
from . import format_utils

class MaskrcnnPred(Predictor):
    def __init__(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config['maskrcnn']['config_file']))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config['maskrcnn']['config_file'])
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = 'cpu'        

        self._model = DefaultPredictor(cfg)

    def predict(self, img_path):
        img = cv2.imread(str(img_path))

        start_time = time.time()
        predictions = self._model(img)
        inference_time = time.time() - start_time

        predictions = self._format_output(predictions, img_path.stem)

        return predictions, inference_time

    def _format_output(self, predictions, img_id):
        return format_utils.detectron_to_coco(predictions, img_id)