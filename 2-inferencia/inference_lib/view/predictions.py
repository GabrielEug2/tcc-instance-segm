from pathlib import Path
import json
import importlib.resources as pkg_resources
from collections.abc import Iterator

import cv2

from . import common_logic

MODEL_MAP_FILE = pkg_resources.path(__package__, 'model_map.json')

# TODO update
def plot(img_files: Iterator[Path], pred_dir:Path, out_dir: Path):
    """
    Raises:
    * `FileNotFoundError`: if any of the required files or dirs doesn't exist.
    * `ValueError`: if some value is not of the correct type
    """
    _test_pred_dir(pred_dir)
    metadata = common_logic.get_metadata(MODEL_MAP_FILE)

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

def _test_pred_dir(pred_dir):
    if not pred_dir.exists():
        raise FileNotFoundError(f"Directory not found: \"{str(pred_dir)}\"")
    if not pred_dir.is_dir():
        raise ValueError(f"{pred_dir} is not a directory")
