
import argparse
import subprocess
import time
import importlib
import os

import cv2

VALID_MODELS = {
    'maskrcnn': {
        'internal_name': 'Mask-RCNN',
        'module_name': 'inference.mask_rcnn'
    },
    'yolact': {
        'internal_name': 'Yolact',
        'module_name': 'inference.yolact'
    },
    'solo': {
        'internal_name': 'SOLO',
        'module_name': 'inference.solo'
    }
}
OUTPUT_DIR = './results'

def build_parser():
    parser = argparse.ArgumentParser(description='Runs instance segmentation on a set of images.')

    parser.add_argument('--models', nargs='*', help='Models to be used.')
    parser.add_argument('--images', nargs='*', help='Images to be segmented.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold to be used. Only predictions with '
                             'a score higher than this value will be kept.')
    parser.add_argument('--wsl', action='store_true',
                        help=("Ignore this if you're using Linux natively. This is "
                              "just a workaround needed to use images from the "
                              "Windows filesystem in Windows Subsystem for Linux (WSL)"))
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    args.models = set(args.models)
    for model in args.models:
        if model not in VALID_MODELS:
            print(f"'{model}' is not a valid model name. Must be one of {VALID_MODELS}.")
            exit()

    if args.wsl:
        # Convert Windows paths to access from WSL
        wslpaths = []
        for img_path in args.images:
            wslpaths.append(subprocess.run(['wslpath', img_path], stdout=subprocess.PIPE).stdout.decode().rstrip('\n'))
        args.images = wslpaths

    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    
    print(f"Running {len(args.images)} images on models: {args.models}...\n")
    for model in args.models:
        module = importlib.import_module(VALID_MODELS[model]['module_name'])
        model_name = VALID_MODELS[model]['internal_name']

        start_time = time.time()
        print(model_name)

        inference_time_on_each_image = []
        i = 1
        for img_path in args.images:
            print(f"  ({i}/{len(args.images)}) {img_path}...", end='', flush=True)

            out_img, inference_time = module.run(img_path, args.threshold)

            img_id, extension = os.path.basename(img_path).split('.')
            out_filename = f"{img_id}_{model_name}.{extension}"
            cv2.imwrite(os.path.join(OUTPUT_DIR, out_filename), out_img)

            inference_time_on_each_image.append(inference_time)

            print(" done")
            i += 1

        average_inference_time = sum(inference_time_on_each_image) / len(args.images)
        elapsed_time = time.time() - start_time

        print()
        print(f"  Done. ({elapsed_time:.3f}s)")
        print(f"  Average inference time: {average_inference_time:.3f}s")