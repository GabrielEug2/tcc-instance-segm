
# https://docs.voxel51.com/user_guide/evaluation.html
# https://docs.voxel51.com/integrations/coco.html?highlight=add_coco_labels

import json
from pathlib import Path

import fiftyone as fo
import fiftyone.utils.coco as fo_coco_utils

from . import conversions

def load_data(dataset_dir, predictions_dir):
    dataset = load_annotations(dataset_dir)
    dataset = load_predictions(dataset, predictions_dir, dataset_dir)

    return dataset

def load_annotations(dataset_dir):
    img_dir = Path(dataset_dir, 'images')
    ann_file = Path(dataset_dir, 'annotations.json')

    fixed_ann_file = ann_file.parent / (ann_file.stem + '_fix.json')
    if fixed_ann_file.exists() == False:
        conversions.fix_class_ids(ann_file)

    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=str(img_dir),
        labels_path=str(fixed_ann_file),
        label_field='ground_truth',
        label_types='segmentations',
        include_id=True
    )

    return dataset

def load_predictions(dataset, predictions_dir, dataset_dir):
    pred_files = list(Path(predictions_dir).glob('*.json'))
    if len(pred_files) == 0:
        print(f"No predictions found on \"{predictions_dir}\".")
        exit()

    ann_file = Path(dataset_dir, 'annotations.json')
    for pred_file in pred_files:
        fixed_pred_file = pred_file.parent / (pred_file.stem + '_fix.json')
        if fixed_pred_file.exists() == False:
            conversions.fix_img_ids(pred_files, ann_file)
            break

    for pred_file in pred_files:
        fixed_pred_file = pred_file.parent / (pred_file.stem + '_fix.json')

        with fixed_pred_file.open('r') as f:
            predictions = json.load(f)

        model_name = pred_file.stem
        fo_coco_utils.add_coco_labels(dataset, model_name, predictions, [])