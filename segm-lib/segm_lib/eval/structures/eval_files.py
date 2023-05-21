
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalFilesForImg:
	custom_anns_dir: Path = None
	custom_preds_dir: Path = None
	coco_anns_file: Path = None
	coco_preds_file: Path = None

@dataclass
class EvalFiles:
	custom_anns_dir: Path = None
	custom_preds_dir: Path = None
	coco_anns_file: Path = None
	coco_preds_file: Path = None
	per_image: dict[str, EvalFilesForImg] = None