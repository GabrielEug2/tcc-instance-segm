
import argparse
import os
import glob
import importlib
import time

MODELS = {
    'maskrcnn': {
        'print_name': 'Mask-RCNN',
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

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)


    print(f"Running on {n_images} images...\n")
    for model in MODELS:
        module = importlib.import_module(MODELS[model]['module_name'])
        model_name = MODELS[model]['print_name']
        print(model_name)

        start_time = time.time()
        inference_time_on_each_image = []
        i = 1
        for img_path in img_paths:
            print(f"  ({i}/{n_images}) {img_path}...", end='', flush=True)

            json_out, inference_time = module.run(img_path)
            # concat all then write only once

            inference_time_on_each_image.append(inference_time)

            print(" done")
            i += 1

        average_inference_time = sum(inference_time_on_each_image) / n_images
        elapsed_time = time.time() - start_time

        print()
        print(f"  Done. ({elapsed_time:.3f}s)")
        print(f"  Average inference time: {average_inference_time:.3f}s\n")