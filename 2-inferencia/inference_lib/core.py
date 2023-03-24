from pathlib import Path
import time
import json

from tqdm import tqdm
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes
from torch import Tensor

from .predictors import MaskrcnnPred, YolactPred, SoloPred
from .format_utils import rle_to_bin_mask

MODELS = [
    {'name': 'maskrcnn', 'predictor': MaskrcnnPred()},
    {'name': 'yolact', 'predictor': YolactPred()},
    {'name': 'solo', 'predictor': SoloPred()}
]

def run_on_all_models(img_dir_str, output_dir_str, save_masks=False):
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
            predictions_file = output_dir / f"{img_path.stem}_{model['name']}_pred.json"
            with predictions_file.open('w') as f:
                json.dump(predictions, f)

            # o plot da imagem
            predictions_img = _plot(predictions, img_path)
            predictions_img_file = output_dir / f"{img_path.stem}_{model['name']}_pred.jpg"
            cv2.imwrite(predictions_img_file, predictions_img)

            # e se quiser, as máscaras
            if save_masks:
                i = 1
                for prediction in predictions:
                    mask = prediction['mask']
                    mask_filename = output_dir / f"{img_path.stem}_{model['name']}_masks" / i
                    cv2.imwrite(mask_filename, rle_to_bin_mask(mask))

            inference_time_on_each_image.append(inference_time)

        elapsed_time = time.time() - start_time
        average_inference_time = sum(inference_time_on_each_image) / n_images
        result_str += f"{model['name']} -- {elapsed_time:.3f}s -- {average_inference_time:.3f}s\n"

    results_file = output_dir / 'time.txt'
    with results_file.open('w') as f:
        f.write(result_str)

def _plot(predictions, img_path):
    classes = []
    scores = []
    masks = []
    boxes = []
    for prediction in predictions:
        classes.append(prediction['class_id'])
        scores.append(prediction['score'])
        boxes.append(prediction['bbox'])
        masks.append(prediction['mask'])

    img = cv2.imread(str(img_path))
    h, w, _ = img.shape
    instances = Instances((h, w))
    instances.pred_boxes = Boxes(Tensor(boxes))
    instances.pred_masks = Tensor(masks)
    instances.pred_classes = Tensor(classes)
    instances.scores = Tensor(scores)

    v = Visualizer(img, MetadataCatalog.get('coco_2017_test'))
    vis_out = v.draw_instance_predictions(instances)
    predictions_img = vis_out.get_image()

    return predictions_img