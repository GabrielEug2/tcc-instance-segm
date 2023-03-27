from pathlib import Path
import time
import datetime
import json

from tqdm import tqdm
import cv2

from .predictors import MaskrcnnPred, YolactPred, SoloPred

MODELS = [
    {'name': 'maskrcnn', 'predictor': MaskrcnnPred()},
    {'name': 'yolact', 'predictor': YolactPred()},
    {'name': 'solo', 'predictor': SoloPred()}
]

def run_on_all_models(img_dir_str, output_dir_str):
    img_dir = Path(img_dir_str)
    output_dir = Path(output_dir_str)

    if not img_dir.exists():
        print(f"Failed to open \"{img_dir_str}\": directory does not exist.")
        exit()
    if not output_dir.exists():
        output_dir.mkdir()

    img_paths = list(img_dir.glob('*.jpg'))
    n_images = len(img_paths)
    if n_images == 0:
        print(f"No images found on \"{img_dir_str}\".")
        exit()
    
    result_str = (f"{n_images} imagens\n"
                  f"{'Modelo'.ljust(8)} {'Tempo total (s)'.ljust(20)} Tempo médio por imagem (s)\n")

    print(f"Running on {n_images} images...\n")
    for model in MODELS:
        predictor = model['predictor']

        print(model['name'])
        start_time = time.time()

        for img_path in tqdm(img_paths):
            img = cv2.imread(str(img_path))
            predictions = predictor.predict(img)

            predictions_file = output_dir / f"{img_path.stem}_{model['name']}_pred.json"
            with predictions_file.open('w') as f:
                json.dump(predictions, f)

        total_time = datetime.timedelta(seconds=(time.time() - start_time))
        average_time = total_time / n_images
        result_str += f"{model['name'].ljust(8)} {str(total_time).ljust(20)} {average_time}\n"

    results_file = output_dir / 'time.txt'
    with results_file.open('w') as f:
        f.write(result_str)
