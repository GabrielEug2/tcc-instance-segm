from pathlib import Path
import time
import datetime
import json
from typing import Union

from tqdm import tqdm
import cv2

from .predictors import MODEL_MAP

def inference(img_dir: str, out_dir: str, models: Union["list[str]", str] = 'all'):
    img_dir = Path(img_dir)
    out_dir = Path(out_dir)
    requested_models = _parse_model_names(models)

    if not img_dir.exists():
        raise FileNotFoundError(f'Directory not found: "{str(img_dir)}"')
    # Não pode ser um generator porque vamos iterar várias vezes
    img_paths = list(img_dir.glob('*.jpg'))
    n_images = len(img_paths)
    if n_images == 0:
        raise FileNotFoundError(f'No images found on "{str(img_dir)}"')

    if not out_dir.exists():
        out_dir.mkdir()

    print(f"Running on {n_images} images...\n")
    inference_stats = {
        'n_images': n_images,
        'model_data': []
    }
    for model_name in requested_models:
        print("\n" + model_name)
        predictor = MODEL_MAP[model_name]

        start_time = time.time()
        for img_path in tqdm(img_paths):
            img = cv2.imread(str(img_path))
            predictions = predictor.predict(img)

            predictions_file = out_dir / f"{img_path.stem}_{model_name}.json"
            with predictions_file.open('w') as f:
                json.dump(predictions, f)
        total_time = datetime.timedelta(seconds=(time.time() - start_time))

        average_time = total_time / n_images
        inference_stats['model_data'].append({
            'name': model_name,
            'total': total_time,
            'average': average_time,
        })

    stats_str = _stats_to_str(inference_stats)    
    stats_file = out_dir / 'stats.txt'
    with stats_file.open('w') as f:
        f.write(stats_str)

def _parse_model_names(model_names):
    VALID_MODELS = MODEL_MAP.keys()

    if model_names == 'all':
        requested_models = VALID_MODELS
    elif type(model_names) == str:
        requested_model = model_names
        if requested_model not in VALID_MODELS:
            raise ValueError(f'Invalid model: "{requested_model}". Must be one of {VALID_MODELS}')
    elif type(model_names) == list:
        requested_models = model_names
        if any(requested_models not in VALID_MODELS):
            raise ValueError(f'Invalid model list: "{requested_models}". Must be a subset of {VALID_MODELS}')
    else:
        raise TypeError('model_names must be list[str] or str')

def _stats_to_str(stats):
    stats_str = (
        f"{stats['n_images']} imagens\n"
        f"{'Modelo'.ljust(10)} {'Tempo total (s)'.ljust(20)} Tempo médio por imagem (s)\n"
    )
    for model in stats['model_data']:
        name = model['name']
        total = model['total']
        average = model['average']
        stats_str += f"{name.ljust(10)} {str(total).ljust(20)} {str(average)}\n"