import json
import os
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances
from pycocotools.coco import COCO
from PySide6.QtGui import QImage, QPixmap

COCO_DIR = 'E:\Desktop\TCC\COCO'
RESULTS_DIR = 'E:\Desktop\TCC\Results'

TEMP_FILENAME = 'temp.jpg'

SPLITS_WITH_ANNOTATIONS = ['train2017', 'val2017']
VALID_SPLITS = ['train2017', 'val2017', 'test2017', 'unlabeled2017']


def load_img(img_path):
    img = cv2.imread(img_path)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb_image


def load_ground_truth(img_path):
    # Se encontrar annotations, plota e retorna a imagem resultante.
    # Se não encontrar, retorna uma imagem preta.

    full_dir, img_filename = os.path.split(img_path)
    data_split = os.path.basename(full_dir)   

    if data_split not in SPLITS_WITH_ANNOTATIONS:
        # Não tá na pasta certa ou é uma imagem sem annotations (ex: do test2017)
        black_img = create_black_img(img_path)

        return black_img

    annotations_filename = 'instances_' + data_split + '.json'

    # Demora 30s pra carregar essa jossa na memória. Vou ter que separar
    # os arquivos de annotations em arquivos menores se quiser mostrar
    # rápido. Mas foda-se, por enquanto fica assim. Tenho outras coisas
    # mais importantes pra fazer no momento.
    # 
    # TODO: 
    # * Em um arquivo separado: 
    #     for each file in valfolder:
    #         img_id = X # extrai do filename
    #         annIds = coco.getAnnIds(img_id)
    #         ann = coco.loadAnns(annIds)
    #         save_ann_ids(output_folder, img_filename)
    #
    # * Depois, com esses arquivos prontos:
    #     annotations = read file
    #     custom_showAnnotations() # copy paste da do COCO
    coco = COCO(os.path.join(COCO_DIR, 'annotations', annotations_filename))

    img_id = int(img_filename.rstrip('.jpg'))
    annIds = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(annIds)

    img = load_img(img_path)

    # ==============================================================
    # Como salvar a imagem com a mesma resolução da imagem original
    # depois de plotar?
    #
    # See:
    #   https://stackoverflow.com/questions/34768717/matplotlib-unable-to-save-image-in-same-resolution-as-original-image
    #   https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size
    # ==============================================================

    height, width, _ = img.shape
    dpi = plt.rcParams['figure.dpi'] # padrão do matlab
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.add_axes([0, 0, 1, 1])

    # Show image
    plt.imshow(img)
    plt.axis('off')

    # Plot something
    coco.showAnns(anns)

    # Save with that same dpi --> output image will have the same size as the original
    fig.savefig(TEMP_FILENAME, dpi=dpi)

    # ==============================================================

    img_with_predictions = load_img(TEMP_FILENAME)
    os.remove(TEMP_FILENAME)

    return img_with_predictions


def load_predictions(img_path):
    # Se encontrar predictions, plota e retorna a imagem resultante.
    # Se não encontrar, retorna uma imagem preta.

    full_dir, img_filename = os.path.split(img_path)
    data_split = os.path.basename(full_dir)

    if data_split not in VALID_SPLITS:
        # Não tá na pasta certa, nem adianta procurar
        black_img = create_black_img(img_path)
        return [black_img, black_img, black_img]

    prediction_files = [
        os.path.join(RESULTS_DIR, data_split, 'Mask-RCNN', img_filename.replace('jpg', 'json')),
        os.path.join(RESULTS_DIR, data_split, 'YOLACT', img_filename.replace('jpg', 'json')),
        os.path.join(RESULTS_DIR, data_split, 'SOLO', img_filename.replace('jpg', 'json'))
    ]
    
    imgs_with_predictions = []

    for filename in prediction_files:
        if os.path.isfile(filename):
            # Tem predictions, plota com a API do Detectron
            with open(filename) as f:
                predictions = json.load(f)

            img = load_img(img_path)

            # Em um mundo ideal eu plotaria direto a partir dos objetos que eu
            # salvei via JSON, mas pra usar a API do Detectron é necessário
            # criar um objeto "Instances".
            instances = Instances(image_size=img.shape[:2])

            pred_masks = torch.tensor(predictions['pred_masks'])
            pred_classes = torch.tensor(predictions['pred_classes'])
            scores = torch.tensor(predictions['scores'])

            instances.set('pred_masks', pred_masks)
            instances.set('pred_classes', pred_classes)
            instances.set('scores', scores)

            v = Visualizer(img, MetadataCatalog.get('coco_2017_test'))
            output = v.draw_instance_predictions(instances)

            output.save(TEMP_FILENAME)

            img_with_predictions = load_img(TEMP_FILENAME)
            os.remove(TEMP_FILENAME)

            imgs_with_predictions.append(img_with_predictions)
        else:
            # Não tem predictions, eu provavelmente não rodei ainda
            black_img = create_black_img(img_path)

            imgs_with_predictions.append(black_img)

    return imgs_with_predictions


def numpy_to_pixmap(img):
    height, width, channels = img.shape
    bytes_per_line = channels * width

    pixmap = QPixmap(QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888))

    return pixmap


def create_black_img(img_path):
    # Retorna uma imagem preta do mesmo tamanho da imagem original
    img = cv2.imread(img_path)
    black_img = np.zeros(img.shape, dtype=np.uint8)

    return black_img