import json

from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances, Boxes
import torch
import numpy as np
import cv2

from .format_utils import rle_to_bin_mask

def get_metadata(map_file):
    with map_file.open('r') as f:
        classmap = json.load(f)

    # PAREI AQUI
    # IDs que foram pulados também precisam de um nome no metadata
    id_name_map = {id: name for name, id in classmap.items()}
    max_id = max(id_name_map.keys())
    for id_ in range(1, max_id):
        id_name_map[id_] = id_name_map.get(id_, None)

    classnames = id_name_map.items()
    sorted_classnames = [x[0] for x in sorted_map]



    sorted_map = sorted(classmap.items(), key=lambda item: item[1])

    metadata = Metadata()
    metadata.set(thing_classes = sorted_classnames)

    return metadata

    
def plot(anns_or_preds, img_path, metadata=None):
    """Plota as annotations ou predictions fornecidas na imagem.

    Args:
        anns_or_preds (list): lista de annotations ou predictions no formato:
            {
                "class_id": int, seguindo a numeração do "metadata",
                "confidence": float, entre 0 e 1, com 1 sendo 100% de certeza,
                "mask": compact RLE, que é a forma que o COCO usa para comprimir máscaras,
                "bbox": [x1, y1, x2, y2],
            }
        img_path (str): caminho para a imagem onde serão plotadas as máscaras
        metadata (detectron2.data.MetadataCatalog): necessário para plotar
            o nome das classes

    Returns:
        np.ndarray: imagem no formato BGR
    """
    classes, scores, masks, boxes = [], [], [], []
    for ann in anns_or_preds:
        classes.append(ann['class_id'] - 1) # Para plotar, precisa estar de [0,N)
        scores.append(ann['confidence'])
        masks.append(rle_to_bin_mask(ann['mask']))
        boxes.append(ann['bbox'])

    img = cv2.imread(str(img_path))
    h, w, _ = img.shape
    instances = Instances((h, w))
    instances.pred_classes = torch.tensor(classes, dtype=torch.int)
    instances.scores = torch.tensor(scores, dtype=torch.float)
    instances.pred_masks = torch.tensor(np.array(masks), dtype=torch.bool)
    instances.pred_boxes = Boxes(torch.tensor(boxes, dtype=torch.float))
       
    v = Visualizer(img, metadata)
    vis_out = v.draw_instance_predictions(instances)
    predictions_img = vis_out.get_image()

    return predictions_img

def _save_masks():
    #     i = 1
    #     for prediction in predictions:
    #         mask = rle_to_bin_mask(prediction['mask'])
    #         mask_filename = output_dir / f"{img_path.stem}_{model['name']}_masks" / f"{i}.jpg"
    #         cv2.imwrite(mask_filename, mask)
    pass