
import argparse
import json
from pathlib import Path

import fiftyone as fo
import fiftyone.utils.openimages as openimages
import fiftyone.zoo as fozoo

COCO_CLASS_DIST_FILE = Path(__file__).parent / 'class_dist_coco.json'

def download(n_imgs, out_dir):
    # Filtra as classes que tem nos dois datasets
    with COCO_CLASS_DIST_FILE.open('r') as f:
        coco_class_dist = json.load(f)

    openimages_classes = [x.lower() for x in openimages.get_segmentation_classes()]
    common_classes = [x for x in coco_class_dist if x in openimages_classes]

    class_dist = {}
    for class_name in common_classes:
        class_dist[class_name] = coco_class_dist.pop(class_name)

    # Pega só as 10 que mais tem no COCO
    classes = sorted(class_dist.items(), key=lambda item: item[1], reverse=True)[:10]
    classes = [x[0].capitalize() for x in classes]

    dataset = fozoo.load_zoo_dataset(
        "open-images-v6",
        split="validation",
        label_types="segmentations",
        classes=classes,
        max_samples=n_imgs
    )
    dataset.export(
        export_dir=out_dir,
        dataset_type=fo.types.COCODetectionDataset
    )
    Path.rename(Path(out_dir / 'data'), Path(out_dir / 'images'))
    Path.rename(Path(out_dir / 'labels.json'), Path(out_dir / 'annotations.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_imgs', type=int, help='Number of images to download', default=200)
    parser.add_argument('-o', '--out_dir', help='Directory to save', default='test_set')

    args = parser.parse_args()

    download(args.n_imgs, args.out_dir)