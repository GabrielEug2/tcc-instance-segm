import os
import glob
import importlib
import time
import json
import warnings
warnings.filterwarnings("ignore") # o Yolact e o SOLO mostram um monte de avisos
                                  # de deprecated, acaba poluindo o terminal

from tqdm import tqdm

from . import maskrcnn, yolact, solo

MODELS = [
    {'name': 'maskrcnn', 'module': maskrcnn},
    {'name': 'yolact', 'module': yolact},
    {'name': 'solo', 'module': solo}
]

def run_on_all_models(input_dir, output_dir):
    if not os.path.isdir(input_dir):
        print(f"Failed to open \"{input_dir}\": directory does not exist.")
        exit()
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    img_paths = glob.glob(input_dir + '/*')
    n_images = len(img_paths)

    result_str = (f"N de imagens: {n_images}\n\n"
                   "Modelo\tTempo total (s)\tTempo médio por imagem (s)\n")

    print(f"Running on {n_images} images...\n")
    for model in MODELS:
        predictions = []
        inference_time_on_each_image = []

        print(model['name'])
        start_time = time.time()

        for i in tqdm(range(n_images)):
            img_path = img_paths[i]

            img_predictions, inference_time = model['module'].run(img_path)

            predictions = [*predictions, *img_predictions]
            inference_time_on_each_image.append(inference_time)

        predictions_file_path = os.path.join(output_dir, f"{model['name']}_pred.json")
        with open(predictions_file_path, 'w') as f:
            json.dump(predictions, f)

        elapsed_time = time.time() - start_time
        average_inference_time = sum(inference_time_on_each_image) / n_images
        result_str += f"{model['name']}\t {elapsed_time:.3f}s\t {average_inference_time:.3f}s\n"

    results_file_path = os.path.join(output_dir, 'tempos.txt')
    with open(results_file_path, 'w') as f:
        f.write(result_str)