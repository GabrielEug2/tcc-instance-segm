import os

from pycocotools.coco import COCO
import matplotlib.pyplot as plt

from segmtools.core import Image
from segmtools.core import utils


COCO_DIR = 'E:\Desktop\TCC\COCO'

_loaded_file = None
_coco_instance = None


def load_annotations(img_path):
    """
    Se tiver annotations, plota e retorna a imagem resultante.
    Caso contrário, retorna uma imagem preta.
    """
    full_dir, img_filename = os.path.split(img_path)
    last_dir = os.path.basename(full_dir)

    if last_dir != 'train2017' and last_dir != 'val2017':
        img = utils.open_as_rgb(img_path)
        black_img = utils.create_black_img(img.shape)
        
        return Image(black_img, "Annotations")


    # Como carregar as annotations na memória demora um pouco, decidi manter uma referência
    # pro objeto, pra não ter que carregar toda vez que a função é chamada.
    global _coco_instance
    global _loaded_file
    annotations_filename = 'instances_' + last_dir + '.json'

    if annotations_filename != _loaded_file:
        _coco_instance = COCO(os.path.join(COCO_DIR, 'annotations', annotations_filename))
        _loaded_file = annotations_filename


    img_id = int(img_filename.rstrip('.jpg'))
    annIds = _coco_instance.getAnnIds(imgIds=img_id)
    anns = _coco_instance.loadAnns(annIds)

    img = utils.open_as_rgb(img_path)
    img_with_annotations = _plot_annotations(anns, img)

    return Image(img_with_annotations, "Annotations")

def load_img(img_path):
    img = utils.open_as_rgb(img_path)

    return Image(img, "Imagem original")



def _plot_annotations(anns, img):
    # ==============================================================
    # Como salvar a imagem com a mesma resolução da imagem original
    # depois de plotar?
    #
    # See:
    #   https://stackoverflow.com/questions/34768717/matplotlib-unable-to-save-image-in-same-resolution-as-original-image
    #   https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size
    # ==============================================================

    # Compute figsize based on imgsize
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
    TEMP_FILENAME = 'test.jpg'
    fig.savefig(TEMP_FILENAME, dpi=dpi)
    plt.close()

    img_with_annotations = utils.open_as_rgb(TEMP_FILENAME)
    os.remove(TEMP_FILENAME)

    return img_with_annotations