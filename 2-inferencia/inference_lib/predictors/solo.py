from pathlib import Path
import time

import cv2
from adet.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from .predictor import Predictor
from .config import config
from . import format_utils

# import warnings
# warnings.filterwarnings("ignore") # o Yolact e o SOLO mostram um monte de avisos
#                                   # de deprecated, acaba poluindo o terminal

class SoloPred(Predictor):
    def __init__(self):
        cfg = get_cfg()
        cfg.merge_from_file(str(Path(config['solo']['dir'], config['solo']['config_file'])))
        cfg.MODEL.WEIGHTS = str(Path(config['solo']['dir'], config['solo']['weights_file']))
        cfg.MODEL.SOLOV2.SCORE_THR = 0.5
        cfg.MODEL.DEVICE = 'cpu'

        self._model = DefaultPredictor(cfg)

    def predict(self, img_path):
        img = cv2.imread(str(img_path))

        start_time = time.time()
        predictions = self._model(img)
        inference_time = time.time() - start_time

        # Por algum motivo além da minha compreensão, o SOLO testa o score
        # de classificação ANTES de ter os scores "definitivos". Isso faz
        # com que ele retorne resultados com score abaixo do que foi solicitado.
        # Pra consertar isso:
        inds = (predictions['instances'].scores > 0.5)
        predictions['instances'] = predictions['instances'][inds]

        predictions = self._format_output(predictions, img_path.stem)

        return predictions, inference_time

    def _format_output(self, predictions, img_id):
        # É igual o do Detectron, foi feito encima dele
        return format_utils.detectron_to_coco(predictions, img_id)