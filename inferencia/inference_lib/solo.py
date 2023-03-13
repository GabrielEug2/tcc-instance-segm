
import os
import time
import sys
sys.path.append('./AdelaiDet/')

import cv2
from adet.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from . import format_utils


SOLO_CONFIG_FILE = "AdelaiDet/configs/SOLOv2/R50_3x.yaml"
SOLO_WEIGHTS = "AdelaiDet/SOLOv2_R50_3x.pth"


def predict(img_path):
    cfg = get_cfg()
    cfg.merge_from_file(SOLO_CONFIG_FILE)
    cfg.MODEL.WEIGHTS = SOLO_WEIGHTS
    cfg.MODEL.SOLOV2.SCORE_THR = 0.5
    cfg.MODEL.DEVICE = 'cpu'

    model = DefaultPredictor(cfg)

    img = cv2.imread(img_path)

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