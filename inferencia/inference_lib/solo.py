from pathlib import Path
import os
import time

import cv2
from adet.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from .config import config
from . import format_utils

def predict(img_path):
    cfg = get_cfg()
    cfg.merge_from_file(str(Path(config['solo']['dir'], config['solo']['config_file'])))
    cfg.MODEL.WEIGHTS = str(Path(config['solo']['dir'], config['solo']['weights_file']))
    cfg.MODEL.SOLOV2.SCORE_THR = 0.5
    cfg.MODEL.DEVICE = 'cpu'

    model = DefaultPredictor(cfg)

    img = cv2.imread(str(img_path))

    start_time = time.time()
    predictions = model(img)
    inference_time = time.time() - start_time

    # Por algum motivo além da minha compreensão, o SOLO testa o score
    # de classificação ANTES de ter os scores "definitivos". Isso faz
    # com que ele retorne resultados com score abaixo do que foi solicitado.
    # Pra consertar isso:
    inds = (predictions['instances'].scores > 0.5)
    predictions['instances'] = predictions['instances'][inds]

    img_id, _ = os.path.basename(img_path).split('.')
    predictions = format_output(predictions, img_id)

    return predictions, inference_time

def format_output(predictions, img_id):
    # O formato é igual o do Detectron, porque foi feito encima dele
    return format_utils.detectron_to_coco(predictions, img_id)