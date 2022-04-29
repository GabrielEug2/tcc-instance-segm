
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def plot(instances, img):
    # o Metadata é pra saber o nome das classes
    v = Visualizer(img, MetadataCatalog.get('coco_2017_test'), scale=1.2)
    vis_out = v.draw_instance_predictions(instances.to('cpu'))

    return vis_out.get_image()
