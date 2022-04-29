
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from .plot import plot


MASK_RCNN_CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"


def run(img_filename, threshold):
    # Construção do modelo
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MASK_RCNN_CONFIG_FILE))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MASK_RCNN_CONFIG_FILE)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
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

    out_img = plot(instances, img)

    return out_img, inference_time