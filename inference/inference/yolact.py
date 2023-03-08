
import time

import cv2
import sys
sys.path.insert(1, './yolact/')
from data import set_cfg
from data import cfg
import torch
from yolact import Yolact
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess
from detectron2.structures import Instances, Boxes

from .plot import plot


YOLACT_CONFIG_FILE = 'yolact_base_config'
YOLACT_WEIGHTS = 'yolact/yolact_base_54_800000.pth'


# Should be similar to:
# https://github.com/dbolya/yolact/issues/256#issuecomment-567371328

def run(img_filename):
    # Criação do modelo
    set_cfg(YOLACT_CONFIG_FILE)
    cfg.mask_proto_debug = False

    model = Yolact()
    model.load_weights(YOLACT_WEIGHTS)
    model.eval()

    # Leitura da imagem
    img = cv2.imread(img_filename)

    # Inferência
    start_time = time.time()

    with torch.no_grad():
        frame = torch.from_numpy(img).float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = model(batch)
        # Essas predições ainda não são finais, falta converter
        # as máscaras pro formato certo.

    h, w, _ = img.shape
    classes, scores, boxes, masks = postprocess(preds, w, h, score_threshold = 0.5)

    inference_time = time.time() - start_time

    # Plot
    instances = Instances((h, w))
    instances.pred_boxes = Boxes(boxes)
    instances.pred_masks = masks
    instances.pred_classes = classes
    instances.scores = scores

    out_img = plot(instances, img)

    return out_img, inference_time
