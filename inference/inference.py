
import argparse
import subprocess
import time
import importlib
import os

import cv2

VALID_MODELS = ['maskrcnn', 'yolact', 'solo']
MODEL_TO_MODULE = {
    'maskrcnn': 'mask_rcnn',
    'yolact': 'yolact',
    'solo': 'solo'
}
MODEL_TO_PRINTABLE_NAME = {
    'maskrcnn': 'MaskRCNN',
    'yolact': 'Yolact',
    'solo': 'SOLO'
}

def build_parser():
    parser = argparse.ArgumentParser(description='Runs instance segmentation on a set of images.')

    parser.add_argument('--images', nargs='*', help='Images to be segmented.')
    parser.add_argument('--models', nargs='*', help='Models to be used.')
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
        for image in args.images:
            wslpaths.append(subprocess.run(['wslpath', image], stdout=subprocess.PIPE).stdout.decode().rstrip('\n'))
        args.image = wslpaths


    print(f"Running {len(args.images)} images on models: {args.models}...\n")
    for model in args.models:
        module = importlib.import_module(MODEL_TO_MODULE[model], package='inference')

        start_time = time.time()
        print(MODEL_TO_PRINTABLE_NAME[model])

        inference_time_on_each_image = []
        for image in args.images:
            print(f"\t{image}...", end='')

            out_img, inference_time = module.run(args.image, args.threshold)

            img_id, extension = os.path.basename(image).split('.')
            out_filename = f"{img_id}_{MODEL_TO_PRINTABLE_NAME[model]}.{extension}"
            cv2.imwrite(os.path.join('results', out_filename), out_img)

            inference_time_on_each_image.append(inference_time)

            print("Done.")

        average_inference_time = sum(inference_time_on_each_image) / len(args.images)
        elapsed_time = time.time() - start_time

        print(f"\n\tDone. ({elapsed_time:.3f}s)")
        print(f"\tAverage inference time: {average_inference_time:.3f}")