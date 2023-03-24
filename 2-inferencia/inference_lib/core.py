from pathlib import Path
import time
import json

from tqdm import tqdm

from .predictors import MaskrcnnPred, YolactPred, SoloPred

MODELS = [
    {'name': 'maskrcnn', 'predictor': MaskrcnnPred()},
    {'name': 'yolact', 'predictor': YolactPred()},
    {'name': 'solo', 'predictor': SoloPred()}
]

def run_on_all_models(img_dir_str, output_dir_str):
    img_dir = Path(img_dir_str)
    output_dir = Path(output_dir_str)

    if not img_dir.is_dir():
        print(f"Failed to open \"{img_dir_str}\": directory does not exist.")
        exit()
    if not output_dir.is_dir():
        output_dir.mkdir()

    img_paths = list(img_dir.glob('*.jpg'))
    n_images = len(img_paths)
    if n_images == 0:
        print(f"No images found on \"{img_dir_str}\".")
        exit()
    
    result_str = (f"{n_images} imagens\n"
                   "Modelo -- Tempo total (s) -- Tempo médio por imagem (s)\n")

    print(f"Running on {n_images} images...\n")
    for model in MODELS:
        predictor = model['predictor']
        inference_time_on_each_image = []

        print(model['name'])
        start_time = time.time()

        for img_path in tqdm(img_paths):
            predictions, inference_time = predictor.predict(img_path)

            # Salva resultados brutos em JSON
            # e o plot da imagem
            predictions_file = output_dir / f"{img_path.stem}_{model['name']}_pred.json"
            with predictions_file.open('w') as f:
                json.dump(predictions, f)

            # plot
            predictions_img = output_dir / f"{img_path.stem}_{model['name']}_pred.jpg"
            # with predictions_file.open('w') as f:
            #     json.dump(predictions, f)

            inference_time_on_each_image.append(inference_time)

        elapsed_time = time.time() - start_time
        average_inference_time = sum(inference_time_on_each_image) / n_images
        result_str += f"{model['name']} -- {elapsed_time:.3f}s -- {average_inference_time:.3f}s\n"

    results_file = output_dir / 'time.txt'
    with results_file.open('w') as f:
        f.write(result_str)