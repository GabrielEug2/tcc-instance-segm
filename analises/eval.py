
# https://docs.voxel51.com/user_guide/evaluation.html
# https://docs.voxel51.com/integrations/coco.html?highlight=add_coco_labels

import argparse
from pathlib import Path
import json

import fiftyone as fo
from fiftyone import ViewField as F

SAMPLE_DATA_DIR = Path(__file__).parent / 'sample_data'

def load_predictions(dataset, pred_files):
    fix_image_ids(dataset)
    load_detections(dataset)
    fix_class_ids(dataset)

def fix_image_ids(dataset):
    print(dataset.first())

    for sample in dataset:
        sample['ground_truth'].image_id = sample.filename
        sample.save()
    
def load_detections(dataset):
    for pred_file in pred_files:
        with pred_file.open('r') as f:
            predictions = json.load(f)
        model_name = pred_file.stem

        for prediction in predictions:
            sample = dataset.match(F("filepath") == prediction.image_id)

            sample[model_name].append(fo.Detection(
                mask=rle_to_bin(prediction.segmentation),
                label=prediction.category_id,
                confidence=prediction.score
            ))

            sample.save()

def fix_class_ids(dataset):

    dataset_class_map = {}
    with COCO_MAP_FILE.open('r') as f:
        coco_class_map = json.load(f)

    # Já o ID das classes é o seguinte: 
    # As classes das predictions seguem a numeração do COCO (pessoa = 1, bike = 2,
    # etc). As annotations baixadas do OpenImages usam uma numeração *diferente*,
    # gerada sequencialmente só com as classes que você baixou, então não dá pra
    # comparar por id. Uma solução seria comparar com o nome, mas assim como o
    # id das imagens que eu mencionei antes, o fiftyone assume que os category_id
    # são int, então achei mais fácil arriscar
    # Teria que usar o nome.

    # muda nas predictions
    # lembra que tem case sensitive no openimages
    # se não tem no coco, marcar como "_OPEX"
    # se não tem no openimages, marcar como "_CCEX"

def rle_to_bin(rle):
    # inverter o processo
    rle_mask = mask.encode(bin_mask.numpy().astype('uint8', order='F'))
    rle_mask['counts'] = rle_mask['counts'].decode('ascii')

    return rle_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', help='directory containing the images and annotations',
                        default=str(SAMPLE_DATA_DIR / 'dataset'))
    parser.add_argument('-p', '--predictions_dir', help='directory containing the predictions',
                        default=str(SAMPLE_DATA_DIR / 'predictions'))

    args = parser.parse_args()

    img_dir = Path(args.dataset_dir, 'images')
    ann_file = Path(args.dataset_dir, 'annotations.json')
    pred_files = Path(args.predictions_dir).glob('*.json')

    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=str(img_dir),
        labels_path=str(ann_file),
        label_field='ground_truth',
        label_type='segmentations',
        include_id=True
    )

    load_predictions(dataset, pred_files)

    session = fo.launch_app(dataset)
    session.wait()