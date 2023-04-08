import json
from pathlib import Path

from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import Metadata
from detectron2.structures import Instances, Boxes
import torch
import numpy as np
import cv2

from .format_utils import rle_to_bin_mask

def get_metadata(map_file: Path) -> Metadata:
	with map_file.open('r') as f:
		classmap = json.load(f)

	# Skipped IDs need a name too, because of the way Visualizer
	# searches for class names (using IDs as indexes in an array
	# of classnames)
	ids = [int(x) for x in classmap.keys()]
	skipped_ids = set(range(0, max(ids))) - set(ids)
	for id_ in skipped_ids:
		classmap[str(id_)] = ''

	sorted_classes = sorted(classmap.items(), key=lambda id_, name: int(id_))
	sorted_classnames = [c[1] for c in sorted_classes]

	metadata = Metadata()
	metadata.set(thing_classes = sorted_classnames)

	return metadata

def plot(anns_or_preds: dict, img_path: Path, metadata: Metadata = None) -> np.ndarray:
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
		classes.append(ann['class_id'])
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