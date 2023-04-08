from pathlib import Path
import json
import importlib.resources as pkg_resources

import cv2

from . import common_logic

CLASSMAP_FILE = pkg_resources.path(__package__, 'classmap_predictions.json')

def plot(img_files: list[Path], pred_dir: Path, out_dir: Path):
    metadata = common_logic.get_metadata(CLASSMAP_FILE)

    if not out_dir.exists():
        out_dir.mkdir()
	
    for img_file in img_files:
        pred_files = list(pred_dir.glob(f"{img_file.stem}_*.json"))
        if len(pred_files) == 0:
            print(f"No predictions found for image {str(img_file)}. Skiping.")
            continue

        for pred_file in pred_files:
            with pred_file.open('r') as f:
                predictions = json.load(f)

            predictions_img = common_logic.plot(predictions, img_file, metadata)

            model_name = pred_file.stem.split('_')[1]
            predictions_img_file = out_dir / f"{img_file.stem}_{model_name}.jpg"
            cv2.imwrite(str(predictions_img_file), predictions_img)