
import argparse
import os
import glob
import importlib
import time
import json
import warnings
warnings.filterwarnings("ignore") # o Yolact e o SOLO mostram um monte de avisos
                                  # de deprecated, acaba poluindo o terminal

from tqdm import tqdm


MODELS = [
    {'name': 'maskrcnn', 'module': 'inference_lib.maskrcnn'},
    {'name': 'yolact', 'module': 'inference_lib.yolact'},
    {'name': 'solo', 'module': 'inference_lib.solo'}
]

def build_parser():
    parser = argparse.ArgumentParser(
        description='Runs instance segmentation on a set of images and store '
                    'the results in the specified folder'
    )

    parser.add_argument('input_dir', help='directory containing the images to segment')
    parser.add_argument('output_dir', help='directory to save the results')

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

    result_str = (f"N de imagens: {n_images}\n\n"
                   "Modelo\tTempo total (s)\tTempo médio por imagem (s)\n")

    print(f"Running on {n_images} images...\n")
    for model in MODELS:
        module = importlib.import_module(model['module'])
        predictions = []
        inference_time_on_each_image = []

        print(model['name'])
        start_time = time.time()

        for i in tqdm(range(n_images)):
            img_path = img_paths[i]

            img_predictions, inference_time = module.run(img_path)

            predictions = [*predictions, *img_predictions]
            inference_time_on_each_image.append(inference_time)

        predictions_file_path = os.path.join(args.output_dir, f"{model['name']}.json")
        with open(predictions_file_path, 'w') as f:
            json.dump(predictions, f)

        elapsed_time = time.time() - start_time
        average_inference_time = sum(inference_time_on_each_image) / n_images
        result_str += f"{model['name']}\t {elapsed_time:.3f}s\t {average_inference_time:.3f}s\n"

    results_file_path = os.path.join(args.output_dir, 'tempos.txt')
    with open(results_file_path, 'w') as f:
        f.write(result_str)
