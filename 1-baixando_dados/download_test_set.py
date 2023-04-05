
import argparse
import json
from pathlib import Path

import fiftyone as fo
import fiftyone.utils.openimages as openimages
import fiftyone.zoo as fozoo

COCO_CLASS_DIST_FILE = Path(__file__).parent / 'class_dist_coco'

def download(n_imgs, out_dir):
    # Filtra as classes que tem nos dois datasets
    with COCO_CLASS_DIST_FILE.open('r') as f:
        coco_class_dist = json.load(f)

    openimages_classes = [x.lower() for x in openimages.get_segmentation_classes()]
    common_classes = [x for x in coco_class_dist if x in openimages_classes]

    common_classes_dist = {}
    for class_name in common_classes:
        common_classes_dist[class_name] = coco_class_dist[class_name]

    # Pega só as 10 que mais tem no COCO
    top_classes = sorted(common_classes_dist.items(), key=lambda item: item[1], reverse=True)[:10]
    top_classes = [x[0].capitalize() for x in top_classes]

    dataset = fozoo.load_zoo_dataset(
        "open-images-v6",
        split="validation",
        label_types="segmentations",
        classes=top_classes,
        max_samples=n_imgs
    )

    out_dir = Path(out_dir)
    dataset.export(
        export_dir=str(out_dir),
        dataset_type=fo.types.COCODetectionDataset
    )
    Path.rename(out_dir / 'data', out_dir / 'images')
    Path.rename(out_dir / 'labels.json', out_dir / 'annotations.json')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_imgs', type=int, help='Number of images to download')
    parser.add_argument('out_dir', help='Directory to save')

    args = parser.parse_args()

    download(args.n_imgs, args.out_dir)