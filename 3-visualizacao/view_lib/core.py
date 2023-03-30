from . import annotations
from . import predictions

def plot_annotations(img_file_or_dir, ann_file, out_dir):
    annotations.plot(img_file_or_dir, ann_file, out_dir)

def plot_predictions(img_file_or_dir, pred_dir, out_dir):
    predictions.plot(img_file_or_dir, pred_dir, out_dir)