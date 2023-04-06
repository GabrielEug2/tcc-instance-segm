from pathlib import Path

from . import annotations
from . import predictions

def plot_predictions(img_file_or_dir: str, pred_dir: str, out_dir: str):
    img_files = _get_img_files(img_file_or_dir)
    out_dir = Path(out_dir)

    predictions.plot(img_files, pred_dir, out_dir)

def plot_annotations(img_file_or_dir: str, ann_file: str, out_dir: str):
    img_files = _get_img_files(img_file_or_dir)
    ann_file = Path(ann_file)
    out_dir = Path(out_dir)
    
    annotations.plot(img_files, ann_file, out_dir)
   
def plot_both(img_file_or_dir: str, ann_file: str, pred_dir: str, out_dir: str):
    img_files = _get_img_files(img_file_or_dir)
    ann_file = Path(ann_file)
    out_dir = Path(out_dir)

    annotations.plot(img_files, ann_file, out_dir)
    predictions.plot(img_files, pred_dir, out_dir)

def _get_img_files(img_file_or_dir: Path) -> "list[Path]":
    img_file_or_dir = Path(img_file_or_dir)

    if not img_file_or_dir.exists():
        raise FileNotFoundError(f"File or dir not found: \"{str(img_file_or_dir)}\"")

    if img_file_or_dir.is_dir():
        img_files = list(img_file_or_dir.glob("*.jpg"))
    else:
        img_files = [img_file_or_dir]

    return img_files

def _save_masks():
    #     i = 1
    #     for prediction in predictions:
    #         mask = rle_to_bin_mask(prediction['mask'])
    #         mask_filename = output_dir / f"{img_path.stem}_{model['name']}_masks" / f"{i}.jpg"
    #         cv2.imwrite(mask_filename, mask)
    pass