
import os
from collections import Counter

import yaml
import cv2
from pycocotools.coco import COCO
import matplotlib.pyplot as plt


with open('config.yaml') as f:
    CONFIG = yaml.safe_load(f)

coco_api = None # Global pra não ter que carregar de novo a cada função


def _load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def _load_annotations(img_path):
    # Usa o diretório da imagem pra encontrar o arquivo certo. Só funciona
    # se a imagem estiver na pasta original (train2017, test2017...)
    full_dir, img_filename = os.path.split(img_path)
    dataset_name = os.path.basename(full_dir)
    annotations_filename = 'instances_' + dataset_name + '.json'

    global coco_api
    coco_api = COCO(os.path.join(CONFIG.ANNOTATIONS_FOLDER, annotations_filename))

    img_id = int(img_filename.rstrip('.jpg'))
    annIds = coco_api.getAnnIds(imgIds=img_id)
    anns = coco_api.loadAnns(annIds)

    return anns

def _plot(annotations, img):
    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    plt.axis('off')

    plt.imshow(img)

    global coco_api
    coco_api.showAnns(annotations)

    # Em vez de mostrar com plt.show(), retorna a figura pro Qt
    fig = plt.gcf()

    # Como essa API não plota o nome das classes, eu só mostro
    # no console quantos objetos de cada classe aparecem na imagem.
    cat_ids = [ann['category_id'] for ann in annotations]
    cats = coco_api.loadCats(cat_ids)
    cat_names = [cat['name'] for cat in cats]
    cat_counts = Counter(cat_names)

    print("\nClasses na imagem:")
    for cat, count in cat_counts.items():
        print(f"  {cat} x{count}")

    return fig

def plot_annotations(img_path):
    img = _load_image(img_path)
    annotations = _load_annotations(img_path)

    matplot_fig = _plot(annotations, img)

    return matplot_fig