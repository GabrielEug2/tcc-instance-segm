from pathlib import Path
import json

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
import cv2

from . import common_logic
from .format_utils import polygon_to_rle

def plot(img_file_or_dir, ann_file, out_dir):
    img_file = Path(img_file_or_dir)
    ann_file = Path(ann_file)
    out_dir = Path(out_dir)

    if img_file_or_dir.is_dir():
        img_files = list(img_file_or_dir.glob("*.jpg"))
        if len(img_files) == 0:
            print(f"No images found on \"{str(img_file_or_dir)}\".")
            exit()
    else:
        img_files = [img_file_or_dir]

    with ann_file.open('r') as f:
        anns = json.load(f)
    metadata = _get_metadata(ann_file)

    for img_file in img_files:
        img_desc = _get_img_desc(img_file, anns)
        if img_desc == None:
            print(f"No description found on \"{str(ann_file)}\" for image \"{str(img_file)}\". Skipping")
            exit()
        
        relevant_anns = _get_annotations(img_desc['id'], anns)
        if len(relevant_anns) == 0:
            print(f"No annotations found on \"{str(ann_file)} for image \"{str(img_file)}\". Skipping")
            continue

        h, w, _ = img_desc['height'], img_desc['width']
        formatted_anns = _to_pred_format(relevant_anns, h, w)
        
        annotated_img = common_logic.plot(formatted_anns, img_file, metadata)
        annotated_img_file = out_dir  / f"{img_file.stem}_groundtruth.jpg"
        cv2.imwrite(str(annotated_img_file), annotated_img)

def _get_metadata(ann_file):
    register_coco_instances('my_dataset', {}, str(ann_file), "")
    metadata = MetadataCatalog.get('my_dataset')

    return metadata

def _get_img_desc(img_file, anns):
    requested_img_desc = None
    for img_desc in anns['images']:
        if img_desc['filename'] == img_file.name:
            requested_img_desc = img_desc
            break
    return requested_img_desc

def _get_annotations(img_id, anns):
    relevant_anns = []

    for ann in anns['annotations']:
        if ann['image_id'] == img_id:
            relevant_anns.append(ann)

    return relevant_anns

def _to_pred_format(anns, h, w):
    formatted_anns = []

    for ann in anns:
        class_id = ann['category_id']
        confidence = 100.0
        mask = polygon_to_rle(ann['segmentation'], h, w)
        bbox = ann['bbox']

        pred = {
            'class_id': class_id,
            'confidence': confidence,
            'mask': mask,
            'bbox': bbox,
        }
        formatted_anns.append(pred)

    return formatted_anns