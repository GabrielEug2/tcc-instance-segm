
import time

import cv2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from .predictor import Predictor
from .config import config
from inference_lib.format_utils import bin_mask_to_rle


class MaskrcnnPred(Predictor):
    def __init__(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config['maskrcnn']['config_file']))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config['maskrcnn']['config_file'])
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = 'cpu'

        self._model = DefaultPredictor(cfg)

    def predict(self, img_path):
        super().predict(img_path)

        img = cv2.imread(str(img_path))

        start_time = time.time()
        raw_predictions = self._model(img)
        inference_time = time.time() - start_time

        # Formato da saída do modelo:
        # https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-format
        predictions = self._to_dict(raw_predictions)

        return predictions, inference_time

    def _to_dict(self, raw_predictions):
        formatted_predictions = []

        instances = raw_predictions['instances']
        for i in range(len(instances)):
            pred_class = instances.pred_classes[i].item()
            score = instances.scores[i].item()
            mask = instances.pred_masks[i]
            bbox = instances.pred_boxes.tensor[i]

            pred = {
                'class_id': pred_class,
                'score': score,
                'mask': bin_mask_to_rle(mask),
                'bbox': bbox,
            }
            formatted_predictions.append(pred)

        return formatted_predictions