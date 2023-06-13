

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalFiles:
	anns_dir: Path = None
	preds_dir: Path = None
	coco_files: dict[str, 'COCOFiles'] = field(default_factory=dict)
	
@dataclass
class COCOFiles:
	anns_file: Path = None
	preds_file: Path = None