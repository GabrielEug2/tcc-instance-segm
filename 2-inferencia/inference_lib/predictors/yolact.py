from pathlib import Path
import time
import sys

import cv2
import torch

from .predictor import Predictor
from .config import config
from inference_lib.format_utils import bin_mask_to_rle

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
        raw_predictions = postprocess(preds, w, h, score_threshold = 0.5)
        inference_time = time.time() - start_time

        # Formato da saída:
        # https://github.com/dbolya/yolact/blob/master/layers/output_utils.py
        predictions = self._to_dict(raw_predictions)

        return predictions, inference_time

    def _to_dict(self, predictions):
        formatted_predictions = []

        for i in range(len(predictions[0])):
            pred_class = predictions[0][i].item()
            score = predictions[1][i].item()
            bbox = predictions[2][i].tolist()
            mask = predictions[3][i].tolist()

            pred = {
                'class_id': pred_class,
                'confidence': score,
                'mask': bin_mask_to_rle(mask),
                'bbox': bbox,
            }
            formatted_predictions.append(pred)

        return formatted_predictions