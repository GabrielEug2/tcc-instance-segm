from pathlib import Path
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

def run_on_all_models(input_dir_str, output_dir_str):
    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)

    if not input_dir.is_dir():
        print(f"Failed to open \"{str(input_dir)}\": directory does not exist.")
        exit()
    if not output_dir.is_dir():
        output_dir.mkdir()

    img_paths = list(input_dir.glob('*.jpg'))
    n_images = len(img_paths)
    if n_images == 0:
        print(f"No images found on \"{str(input_dir)}\".")
        exit()
    
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

            img_predictions, inference_time = model['module'].predict(img_path)

            predictions = [*predictions, *img_predictions]
            inference_time_on_each_image.append(inference_time)

        predictions_file_path = output_dir / f"{model['name']}_pred.json"
        with predictions_file_path.open('w') as f:
            json.dump(predictions, f)

        elapsed_time = time.time() - start_time
        average_inference_time = sum(inference_time_on_each_image) / n_images
        result_str += f"{model['name']}\t {elapsed_time:.3f}s\t {average_inference_time:.3f}s\n"

    results_file_path = output_dir / 'tempos.txt'
    with results_file_path.open('w') as f:
        f.write(result_str)