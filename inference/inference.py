
import argparse
import os
import glob
import importlib
import time

import cv2

MODEL_INFO = {
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

def build_parser():
    parser = argparse.ArgumentParser(description='Runs instance segmentation on a set of images. By default, uses Mask RCNN.')

    parser.add_argument('input_dir', help='directory containing the images to segment')
    parser.add_argument('output_dir', help='directory to save the output of the models')
    parser.add_argument('-y', '--yolact', action='store_true', help='run only on Yolact')
    parser.add_argument('-s', '--solo', action='store_true', help='run only on SOLO')
    parser.add_argument('-a', '--all', action='store_true', help='runs on all available models')
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                        help='Confidence threshold to be used. Only predictions with '
                             'a score higher than this value will be kept. Defaults to 0.5')
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Failed to open \"{args.input_dir}\": directory does not exist.")
        exit()

    img_paths = glob.glob(args.input_dir + '/*')
    n_images = len(img_paths)
    if n_images == 0:
        print(f"No images found on {args.input_dir}")
        exit()

    models = ['maskrcnn']
    if args.yolact:
        models = ['yolact']
    if args.solo:
        models = ['solo']
    if args.all:
        models = ['maskrcnn', 'yolact', 'solo']

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    print(f"Running {n_images} images on models: {models}...\n")
    for model in models:
        module = importlib.import_module(MODEL_INFO[model]['module_name'])
        model_name = MODEL_INFO[model]['internal_name']

        start_time = time.time()
        print(model_name)

        inference_time_on_each_image = []
        i = 1
        for img_path in img_paths:
            print(f"  ({i}/{n_images}) {img_path}...", end='', flush=True)

            out_img, inference_time = module.run(img_path, args.threshold)

            img_id, extension = os.path.basename(img_path).split('.')
            out_filename = f"{img_id}_{model_name}.{extension}"
            cv2.imwrite(os.path.join(args.output_dir, out_filename), out_img)

            inference_time_on_each_image.append(inference_time)

            print(" done")
            i += 1

        average_inference_time = sum(inference_time_on_each_image) / n_images
        elapsed_time = time.time() - start_time

        print()
        print(f"  Done. ({elapsed_time:.3f}s)")
        print(f"  Average inference time: {average_inference_time:.3f}s\n")