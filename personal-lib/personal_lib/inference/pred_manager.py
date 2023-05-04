import json
from pathlib import Path

from personal_lib.core import mask_conversions

class PredManager:
	def __init__(self, root_dir: Path):
		self.root_dir = root_dir
		
	def save(self, preds: dict, img_file_name: str, model_name: str):
		for pred in preds:
			pred['mask'] = mask_conversions.bin_mask_to_rle(pred['mask'])

		out_file = self.root_dir / model_name / f"{img_file_name}.json"
		out_file.parent.mkdir(parents=True, exist_ok=True)
		with out_file.open('w') as f:
			json.dump(preds, f)