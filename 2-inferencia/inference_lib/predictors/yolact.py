from pathlib import Path
import time
import sys

import cv2
import torch

from .predictor import Predictor
from . import format_utils
from ..config import config

sys.path.insert(0, config['yolact']['dir'])
from data import set_cfg
from data import cfg
from yolact import Yolact as YolactLib
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess


class YolactPred(Predictor):
    # Based on:
    # https://github.com/dbolya/yolact/issues/256#issuecomment-567371328

    def __init__(self):
        set_cfg(config['yolact']['config_name'])
        cfg.mask_proto_debug = False

        model = YolactLib()
        model.load_weights(str(Path(config['yolact']['dir'], config['yolact']['weights_file'])))
        model.eval()

        self._model = model
        
    def predict(self, img_path):
        img = cv2.imread(str(img_path))

        start_time = time.time()
        with torch.no_grad():
            frame = torch.from_numpy(img).float()
            batch = FastBaseTransform()(frame.unsqueeze(0))
            preds = self._model(batch)
            # Essas predições ainda não são finais, falta converter
            # as máscaras pro formato certo.
        h, w, _ = img.shape
        classes, scores, _, masks = postprocess(preds, w, h, score_threshold = 0.5)
        predictions = {'classes': classes, 'scores': scores, 'masks': masks}
        inference_time = time.time() - start_time

        predictions = self._format_output(predictions, img_path.stem)

        return predictions, inference_time

    def _format_output(self, predictions, img_id):   
        # Model output - https://github.com/dbolya/yolact/blob/master/layers/output_utils.py
        # COCO format - https://cocodataset.org/#format-results
        coco_style_predictions = []

        for i in range(len(predictions['masks'])):
            pred_class = predictions['classes'][i].item()
            score = predictions['scores'][i].item()
            bin_mask = predictions['masks'][i]

            coco_style_prediction = format_utils.to_coco_format(img_id, pred_class, score, bin_mask)

            coco_style_predictions.append(coco_style_prediction)

        return coco_style_predictions