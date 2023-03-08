
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import cv2

def plot(instances, img):
    # o Metadata é pra saber o nome das classes
    v = Visualizer(img, MetadataCatalog.get('coco_2017_test'), scale=1.2)
    vis_out = v.draw_instance_predictions(instances)

    return vis_out.get_image()

img_id, extension = os.path.basename(img_path).split('.')
out_filename = f"{img_id}_{model_name}.{extension}"

cv2.imwrite(os.path.join(args.output_dir, out_filename), out_img)