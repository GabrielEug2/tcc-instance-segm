
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import torch
from adet.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures import Boxes

from .plot import plot


SOLO_CONFIG_FILE = "AdelaiDet/configs/SOLOv2/R50_3x.yaml"
SOLO_WEIGHTS = "AdelaiDet/SOLOv2_R50_3x.pth"


def run(img_filename, threshold):
    # Construção do modelo
    cfg = get_cfg()
    cfg.merge_from_file(SOLO_CONFIG_FILE)
    cfg.MODEL.WEIGHTS = SOLO_WEIGHTS
    cfg.MODEL.SOLOV2.SCORE_THR = threshold
    cfg.MODEL.DEVICE = 'cpu'

    model = DefaultPredictor(cfg)

    # Leitura da imagem
    img = cv2.imread(img_filename)

    # Inferência
    start_time = time.time()
    predictions = model(img)
    inference_time = time.time() - start_time

    # Plot
    instances = predictions['instances']

    # Por algum motivo além da minha compreensão, o SOLO compara os
    # scores com os thresholds ANTES de ter os scores "definitivos".
    # Isso faz com que ele retorne resultados abaixo do que foi 
    # solicitado. Pra consertar isso:
    inds = (instances.scores > threshold)
    instances = instances[inds]

    # O SOLO prediz as máscaras direto, sem calcular bounding boxes.
    # O único problema disso é que o nome das classes no plot fica
    # tudo empilhado no canto. Felizmente eles tem um código pra
    # computar as bounding boxes a partir das máscaras.
    pred_boxes = torch.zeros(instances.pred_masks.size(0), 4)
    for i in range(instances.pred_masks.size(0)):
       mask = instances.pred_masks[i].squeeze()
       ys, xs = torch.where(mask)
       pred_boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()]).float()
    instances.pred_boxes = Boxes(pred_boxes)

    out_img = plot(instances, img)

    return out_img, inference_time