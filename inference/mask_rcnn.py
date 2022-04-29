
import argparse
import subprocess
import os
import time

import cv2
import yaml
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from plot import plot


with open('config/test_config.yaml') as f:
    TEST_CONFIG = yaml.safe_load(f)

MASK_RCNN_CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"


def build_parser():
    parser = argparse.ArgumentParser(description='Instance segmentation using Mask R-CNN.')
    parser.add_argument('image', type=str, help='Path of an image to be segmented.')
    parser.add_argument('--wsl', action='store_true',
                        help=("Ignore this if you're using Linux natively. This is "
                               "just a workaround needed to use images from the "
                               "Windows filesystem in Windows Subsystem for Linux (WSL)"))
    return parser

def run_on_maskrcnn(img_filename):
    # Construção do modelo
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MASK_RCNN_CONFIG_FILE))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MASK_RCNN_CONFIG_FILE)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = TEST_CONFIG['SCORE_THRESH']
    
    cfg.MODEL.DEVICE = 'cpu'

    model = DefaultPredictor(cfg)
    
    # Leitura da imagem
    img = cv2.imread(img_filename)

    # Inferência
    start_time = time.time()
    predictions = model(img)
    inference_time = time.time() - start_time

    # Plot
    instances = predictions['instances']

    out_img = plot(instances, img)
    full_time = time.time() - start_time

    return out_img, inference_time, full_time


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    if args.wsl:
        # Convert path to access from WSL
        linux_path = subprocess.run(['wslpath', args.image], stdout=subprocess.PIPE).stdout.decode().rstrip('\n')
        args.image = linux_path

    print("Running on Mask RCNN...\n")
    out_img, inference_time, full_time = run_on_maskrcnn(args.image)

    print(f"\nDone. (inference: {inference_time:.3f}s -- with plot: {full_time:.3f})")
    print()

    if args.wsl:
        # Can't show because there's no display, so we save results instead
        img_id, extension = os.path.basename(linux_path).split('.')
        out_filename = os.path.join('results', img_id + '_MaskRCNN' + extension)
        cv2.imwrite(out_filename, out_img)
        exit()

    cv2.imshow('Saída do Mask R-CNN', out_img)