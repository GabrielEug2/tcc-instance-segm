import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from pycocotools.coco import COCO
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances
import torch
import time

from segmtools.core import Image


COCO_DIR = 'E:\Desktop\TCC\COCO'
TEMP_FILENAME = 'temp.jpg'

_MASK_RCNN_FILENAME = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

_loaded_file = None
_coco_instance = None


def load_img(img_path):
    """
    Retorna a imagem especificada no formato certo pro ImgViewer.
    """
    img = _open_as_rgb(img_path)

    return Image(img, "Imagem original")

def load_ground_truth(img_path):
    """
    Se tiver annotations, plota e retorna a imagem resultante.
    Caso contrário, retorna uma imagem preta.
    """
    full_dir, img_filename = os.path.split(img_path)
    last_dir = os.path.basename(full_dir)
    data_split = last_dir

    if data_split not in ['train2017', 'val2017']:
        # Não tá na pasta certa ou é uma imagem sem annotations (ex: do test2017)
        img = cv2.imread(img_path)
        black_img = _create_black_img(img.shape)
        
        return Image(black_img, "Annotations")

    annotations_filename = 'instances_' + data_split + '.json'

    global _coco_instance
    global _loaded_file
    if annotations_filename != _loaded_file:
        # Primeira vez rodando ou mudou de pasta -> carrega o arquivo novo na memória.
        # É melhor carregar uma vez só do que toda vez que chama a função. O do val2017
        # é rápido, mas o do train2017 leva uns 30 segundos.
        _coco_instance = COCO(os.path.join(COCO_DIR, 'annotations', annotations_filename))
        _loaded_file = annotations_filename

    img_id = int(img_filename.rstrip('.jpg'))
    annIds = _coco_instance.getAnnIds(imgIds=img_id)
    anns = _coco_instance.loadAnns(annIds)

    img = _open_as_rgb(img_path)

    # ==============================================================
    # Como salvar a imagem com a mesma resolução da imagem original
    # depois de plotar?
    #
    # See:
    #   https://stackoverflow.com/questions/34768717/matplotlib-unable-to-save-image-in-same-resolution-as-original-image
    #   https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size
    # ==============================================================

    img_height, img_width, _ = img.shape
    dpi = plt.rcParams['figure.dpi'] # padrão do matlab
    fig_height = img_height / float(dpi)
    fig_width = img_width / float(dpi)
    figsize = (fig_width, fig_height)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.add_axes([0, 0, 1, 1])

    # Show image
    plt.imshow(img)
    plt.axis('off')

    # Plot something
    _coco_instance.showAnns(anns)

    # Save with that same dpi --> output image will have the same size as the original
    fig.savefig(TEMP_FILENAME, dpi=dpi)
    plt.close()

    # ==============================================================

    gt_img = _open_as_rgb(TEMP_FILENAME)
    os.remove(TEMP_FILENAME)

    return Image(gt_img, "Annotations")

def run_on_all_models(img_path):
    """
    Roda a imagem especificada em todos os modelos e retorna as imagens
    resultantes em um array.
    """
    img = cv2.imread(img_path)
    predictions = []

    # =====================================================
    # Mask R-CNN
    # =====================================================
    mask_rcnn = _build_maskrcnn()

    start_time = time.time()
    output = mask_rcnn(img)
    inference_time = time.time() - start_time

    img_with_predictions = _maskrcnn_to_img(output, img)

    predictions.append(Image(img_with_predictions, f"Predictions do Mask R-CNN ({inference_time:.3f}s de inferência)"))

    # =====================================================
    # YOLACT
    # =====================================================
    
    # =====================================================
    # SOLO
    # =====================================================

    return predictions


def _open_as_rgb(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def _create_black_img(shape):
    black_img = np.zeros(shape, dtype=np.uint8)
    return black_img

def _build_maskrcnn():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(_MASK_RCNN_FILENAME))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(_MASK_RCNN_FILENAME)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cpu'

    model = DefaultPredictor(cfg)

    return model

def _maskrcnn_to_img(raw_output, img):
    img = _plot_with_visualizer(raw_output['instances'], img)

    return img

def _yolact_to_img(raw_output, img):
    # Formato de saída do YOLACT:
    #   xxx
    
    # yolact to tensors

    # instances object
    # instances = Instances(image_size=img.shape[:2])
    # instances.set('pred_masks', pred_masks)
    # instances.set('pred_classes', pred_classes)
    # instances.set('scores', scores)

    # img = _plot_with_visualizer(instances, img)

    # return img

    pass

def _solo_to_img(raw_output, img):    
    pass

def _plot_with_visualizer(instances, img):
    """
    Plota as predições na imagem e retorna a imagem (como um np.array).

    Args:
        instances: `Instances` object (see detectron2.structures.instances) com as predições.
        img: imagem na qual serão plotadas as instâncias.
    """
    v = Visualizer(img, MetadataCatalog.get('coco_2017_test'))
    vis_output = v.draw_instance_predictions(instances)

    # A figura do matplotlib não me serve pra nada, eu preciso de uma
    # imagem. Felizmente eles já implementaram a conversão.
    vis_output.save(TEMP_FILENAME)

    img = _open_as_rgb(TEMP_FILENAME)
    os.remove(TEMP_FILENAME)

    return img