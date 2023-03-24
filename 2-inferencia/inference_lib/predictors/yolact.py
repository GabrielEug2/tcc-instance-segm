from pathlib import Path

import torch

from .base_pred import BasePred
from .config import config
from inference_lib.format_utils import bin_mask_to_rle

import sys
sys.path.insert(0, config['yolact']['dir'])
from data import set_cfg
from data import cfg
from yolact import Yolact as YolactLib
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess


class YolactPred(BasePred):
    # Based on:
    # https://github.com/dbolya/yolact/issues/256#issuecomment-567371328

    def __init__(self):
        set_cfg(config['yolact']['config_name'])
        cfg.mask_proto_debug = False

        model = YolactLib()
        model.load_weights(str(Path(config['yolact']['dir'], config['yolact']['weights_file'])))
        model.eval()

        self._model = model
        
    def predict(self, img):
        raw_predictions = self._inference(img)

        formatted_predictions = self._to_custom_format(raw_predictions)
        return formatted_predictions

    def _inference(self, img):
        with torch.no_grad():
            frame = torch.from_numpy(img).float()
            batch = FastBaseTransform()(frame.unsqueeze(0))
            preds = self._model(batch)
            # Essas predições ainda não são finais, falta converter
            # as máscaras pro formato certo.
        h, w, _ = img.shape
        raw_predictions = postprocess(preds, w, h, score_threshold = 0.5)

        return raw_predictions

    def _to_custom_format(self, predictions):
        # Formato da saída do modelo:
        #   https://github.com/dbolya/yolact/blob/master/layers/output_utils.py
        #
        # Formato esperado:
        #   see inference_lib.predictors.base_pred

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
            # TODO: inspect mask
            formatted_predictions.append(pred)

        return formatted_predictions