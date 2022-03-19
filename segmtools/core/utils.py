import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from PySide6.QtGui import QImage, QPixmap

COCO_DIR = 'E:\Desktop\TCC\COCO'

def load_img(img_path):
    img = cv2.imread(img_path)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb_image

def load_ground_truth(img_path):
    # Se encontrar annotations, plota e retorna a imagem resultante.
    # Se não encontrar, retorna uma imagem preta.

    full_dir, img_filename = os.path.split(img_path)
    data_split = os.path.basename(full_dir)   

    if data_split == 'val2017' or data_split == 'train2017':
        segmentation_file = 'instances_' + data_split + '.json'

        # Demora 30s pra carregar essa jossa na memória. Vou ter que separar
        # os arquivos de annotations em arquivos menores se quiser mostrar
        # rápido. Mas foda-se, por enquanto fica assim. Tem outras coisas mais
        # importantes no momento.
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
        coco = COCO(os.path.join(COCO_DIR, 'annotations', segmentation_file))

        img_id = int(img_filename.rstrip('.jpg'))
        annIds = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(annIds)

        img = load_img(img_path)

        # ==============================================================
        # https://stackoverflow.com/questions/34768717/matplotlib-unable-to-save-image-in-same-resolution-as-original-image
        # https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size
        # ==============================================================

        # What size does the figure need to be in inches to fit the image?
        #
        #   dpi * inches = n_pixels
        #
        # logo...
        #
        #   inches = n_pixels * dpi
        #
        # Eu tenho o número de pixels (img.shape), é só fixar um dpi e
        # calcular o tamanho em inches.
        height, width, _ = img.shape
        dpi = plt.rcParams['figure.dpi'] # padrão do matlab
        figsize = width / float(dpi), height / float(dpi)

        # Create a figure of the right size with one axes that takes up the full figure
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])

        # Hide spines, ticks, etc.
        ax.axis('off')

        # Show image
        plt.imshow(img)
        plt.axis('off')

        # Plot something
        coco.showAnns(anns)

        fig.savefig('test.jpg', dpi=dpi)

        # ==============================================================

        img_with_segmentations = load_img('test.jpg')

        return img_with_segmentations
    else:
        # Não tá na pasta certa ou é uma imagem sem annotations
        img = cv2.imread(img_path)
        black_img = np.zeros(img.shape, dtype=np.uint8)

        return black_img

def load_predictions(img_path):
    return [[], [], []]

def numpy_to_pixmap(img):
    height, width, channels = img.shape
    bytes_per_line = channels * width
    pixmap = QPixmap(QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888))
    return pixmap


# =====================================================================
# TO DO: plotar o resultado com a própria API do Detectron
# =====================================================================

# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog

# test_img = os.listdir(OUTPUT_DIR):
# img_filename = os.path.basename(results_file).rstrip('.json') + '.jpg'

# img = cv2.imread(os.path.join(IMG_DIR, filename) # BGR


# base.results_file

# with open(filename) as f:
#     predictions = json.load(f)

# img_filename
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# v = Visualizer(img, MetadataCatalog.get('teste'), scale=1.2)
# v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
