
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalFiles:
	filtered_anns_dir: Path = None
	filtered_preds_dir: Path = None
	filtered_coco_anns_file: Path = None
	filtered_coco_preds_file: Path = None
	per_image: dict[str, 'EvalFilesForImg'] = field(default_factory=dict)

@dataclass
class EvalFilesForImg:
	filtered_coco_anns_file: Path = None
	filtered_coco_preds_file: Path = None