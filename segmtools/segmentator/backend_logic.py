import time
import os
import json
import subprocess
from enum import Enum

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
import torch
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from segmtools.core import Image
from segmtools.core import utils


SCORE_THRESH_TEST = 0.5
MASK_RCNN_CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
YOLACT_SCRIPT_FILE = 'models/yolact/eval.py'
YOLACT_WEIGHTS_FILE = 'models/yolact/yolact_base_54_800000.pth'


class Model(Enum):
    MASK_RCNN = 1
    YOLACT = 2
    SOLO = 3


def run_on_all_models(img_path):
    """
    Roda a imagem especificada em todos os modelos e retorna as imagens
    resultantes em um array.
    """
    img = utils.open_as_rgb(img_path)
    results = []

    description_template = ("Predições do {model}\n"
                         "{inference_time:.3f}s de inferência – {n_instances} objetos encontrados")

    # =====================================================
    # Mask R-CNN
    # =====================================================

    print("Running Mask RCNN...")
    start_time = time.time()
    raw_predictions = _run_on_maskrcnn(img)
    inference_time = time.time() - start_time
    print(f"Done {inference_time}")

    print("Plotting...")
    start_time = time.time()
    resulting_img, n_instances = _plot_predictions(raw_predictions, img, Model.MASK_RCNN)
    plotting_time = time.time() - start_time
    print(f"Done {plotting_time}\n")

    description = description_template.format(model='Mask R-CNN',
                                              inference_time=inference_time,
                                              n_instances=n_instances)

    results.append(Image(resulting_img, description))

    # =====================================================
    # YOLACT
    # =====================================================
    
    print("Running YOLACT...")
    start_time = time.time()
    raw_predictions = _run_on_yolact(img_path)
    inference_time = time.time() - start_time
    print(f"Done {inference_time}")

    print("Plotting...")
    start_time = time.time()
    resulting_img, n_instances = _plot_predictions(raw_predictions, img, Model.YOLACT)
    plotting_time = time.time() - start_time
    print(f"Done {plotting_time}\n")

    description = description_template.format(model='YOLACT',
                                              inference_time=inference_time,
                                              n_instances=n_instances)

    results.append(Image(resulting_img, description))
    print("\nDone")

    # =====================================================
    # SOLO
    # =====================================================

    return results

def load_img(img_path):
    img = utils.open_as_rgb(img_path)

    return Image(img, "Imagem original")



def _run_on_maskrcnn(img):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MASK_RCNN_CONFIG_FILE))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MASK_RCNN_CONFIG_FILE)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST
    cfg.MODEL.DEVICE = 'cpu'

    model = DefaultPredictor(cfg)

    output = model(img)

    return output

def _run_on_yolact(img_path):
    subprocess.run(["python", YOLACT_SCRIPT_FILE, f"--trained_model={YOLACT_WEIGHTS_FILE}",
                    f"--score_threshold={SCORE_THRESH_TEST}", f"--image={img_path}"])

    with open('temp.json') as f:
        output = json.load(f)

    return output

def _run_on_solo(img):
    pass



def _plot_predictions(predictions, img, predictions_format:Model):
    """
    Plota as predições na imagem e retorna a imagem (como um np.array).
    """
    # Converte a saída pro formato usado no Visualizer
    match predictions_format:
        case Model.MASK_RCNN:
            instances = _maskrcnn_to_instances(predictions)

        case Model.YOLACT:
            instances = _yolact_to_instances(predictions, img.shape)

        case Model.SOLO:
            instances = _solo_to_instances(predictions)

    # O Metadata é só pra saber o nome das classes
    v = Visualizer(img, MetadataCatalog.get('coco_2017_test'))
    vis_output = v.draw_instance_predictions(instances)

    # A figura do matplotlib não me serve pra nada, eu preciso de uma
    # imagem. Felizmente eles já implementaram a conversão.
    TEMP_FILENAME = 'test.jpg'
    vis_output.save(TEMP_FILENAME)
    img = utils.open_as_rgb(TEMP_FILENAME)
    os.remove(TEMP_FILENAME)

    n_instances = len(instances)

    return img, n_instances

def _maskrcnn_to_instances(predictions):
    instances = predictions['instances']

    return instances

def _yolact_to_instances(predictions, img_shape):
    masks = torch.tensor(predictions['masks'], dtype=torch.uint8)
    classes = torch.tensor(predictions['classes'], dtype=torch.uint8)
    scores = torch.tensor(predictions['scores'], dtype=torch.float32)

    # instances object
    instances = Instances(image_size=img_shape[:2])
    instances.set('pred_masks', masks)
    instances.set('pred_classes', classes)
    instances.set('scores', scores)

    return instances

def _solo_to_instances(predictions):
    return