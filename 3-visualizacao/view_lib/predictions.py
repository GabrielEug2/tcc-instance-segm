from pathlib import Path
import json
import __main__

from detectron2.data import MetadataCatalog
import cv2

from . import common_logic


MODEL_MAP_FILE = Path(__main__.__file__).parent / 'model_map.json'


def plot(img_file_or_dir, pred_dir, out_dir):
    img_file_or_dir = Path(img_file_or_dir)
    pred_dir = Path(pred_dir)
    out_dir = Path(out_dir)

    if img_file_or_dir.is_dir():
        img_files = list(img_file_or_dir.glob("*.jpg"))
        if len(img_files) == 0:
            print(f"No images found on \"{str(img_file_or_dir)}\".")
            exit()
    else:
        img_files = [img_file_or_dir]

    metadata = common_logic.get_metadata(MODEL_MAP_FILE)

    if not out_dir.exists():
        out_dir.mkdir()

    for img_file in img_files:
        pred_files = list(pred_dir.glob(f"{img_file.stem}_*.json"))
        if len(pred_files) == 0:
            print(f"No predictions found on \"{str(pred_dir)}\" for image {str(img_file)}. Skiping.")
            continue

        for pred_file in pred_files:
            with pred_file.open('r') as f:
                predictions = json.load(f)

            predictions_img = common_logic.plot(predictions, img_file, metadata)
            model_name = pred_file.stem.split('_')[1]
            predictions_img_file = out_dir / f"{img_file.stem}_{model_name}.jpg"
            cv2.imwrite(str(predictions_img_file), predictions_img)