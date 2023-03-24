
import time

import cv2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from .config import config
from .predictor import Predictor
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
        pass
        # return instances object


        # coco_style_predictions = []

        # for i in range(len(predictions['instances'])):
        #     pred_class = predictions['instances'].pred_classes[i].item()
        #     score = predictions['instances'].scores[i].item()
        #     bin_mask = predictions['instances'].pred_masks[i]

        #     coco_style_prediction = to_coco_format(img_id, pred_class, score, bin_mask)

        #     coco_style_predictions.append(coco_style_prediction)

        # return coco_style_predictions