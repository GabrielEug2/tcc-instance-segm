
# https://docs.voxel51.com/user_guide/evaluation.html
# https://docs.voxel51.com/integrations/coco.html?highlight=add_coco_labels

import json
from pathlib import Path

import fiftyone as fo
from fiftyone import ViewField as F
# import fiftyone.utils.coco as fo_coco_utils

from . import conversions

REL_IMG_PATH = 'images'
REL_ANN_PATH = 'annotations'
NORMAL_ANN_SUFIX = '.json'
FIXED_ANN_SUFIX = '_fix.json'

NORMAL_PRED_SUFFIX = '.json'
FIXED_PRED_SUFIX = '_fix.json'

def load_data(dataset_dir, predictions_dir):
    dataset = load_annotations(dataset_dir)
    dataset = load_predictions(dataset, predictions_dir, dataset_dir)

    return dataset

def load_annotations(dataset_dir):
    img_dir = Path(dataset_dir, REL_IMG_PATH)
    fixed_ann_file = Path(dataset_dir, REL_ANN_PATH + FIXED_ANN_SUFIX)

    if not fixed_ann_file.exists():
        raw_ann_file = Path(dataset_dir, REL_ANN_PATH + NORMAL_ANN_SUFIX)
        conversions.fix_annotations(raw_ann_file)

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
    raw_pred_files = []
    for file in Path(predictions_dir).glob('*.json'):
        if not file.name.endswith(FIXED_ANN_SUFIX):
            raw_pred_files.append(file)

    if len(raw_pred_files) == 0:
        print(f"No predictions found on \"{predictions_dir}\".")
        exit()

    fixed_pred_files = []
    for raw_pred_file in raw_pred_files:
        fixed_pred_files.append(Path(
            predictions_dir,
            raw_pred_file.name.removesuffix(NORMAL_PRED_SUFFIX) + FIXED_PRED_SUFIX
        ))

    if any(not f.exists() for f in fixed_pred_files):
        fixed_ann_file = Path(dataset_dir, REL_ANN_PATH + FIXED_ANN_SUFIX)
        conversions.fix_predictions(raw_pred_files, fixed_ann_file)

    classes = dataset.default_classes
    id_detections_map = {}
    for pred_file in fixed_pred_files:
        with pred_file.open('r') as f:
            predictions = json.load(f)

        model_name = pred_file.name.removesuffix(FIXED_PRED_SUFIX)
        for prediction in predictions:
            det_list = id_detections_map.get(prediction['image_id'], [])
            det_list.append(
                fo.Detection(
                    mask=conversions.rle_to_bin(prediction['segmentation']),
                    label=classes[prediction['category_id']],
                    confidence=prediction['score']
                )
            )
            id_detections_map[prediction['image_id']] = det_list

        for sample in dataset:
            # sample = dataset.match(F("ground_truth_coco_id") == prediction['image_id']).first()
            coco_id = sample.ground_truth_coco_id
            sample[model_name] = fo.Detections(detections=id_detections_map[coco_id])
            sample.save()

        # fo_coco_utils.add_coco_labels(
        #     dataset,
        #     model_name,
        #     predictions,
        #     dataset.default_classes,
        #     label_type='segmentations',
        #     coco_id_field='ground_truth_coco_id'
        # )

    return dataset