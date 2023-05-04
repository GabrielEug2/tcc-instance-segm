"""Functions to work with predictions in personal_lib format.
See personal_lib.inference.predictors.base_predictor and
personal_lib.inference.pred_manager.save() for more details"""

import json
from pathlib import Path
import shutil

from . import mask_conversions

class Predictions:
	def __init__(self, pred_dir: Path):
		if not pred_dir.exists():
			raise FileNotFoundError(f"Dir not found {str(pred_dir)}")

		self.root_dir = pred_dir

		# could probably optimize for more images keeping the generator
		# but in my case it doesn't really matter
		pred_files = list(pred_dir.glob('*.json'))

		if len(pred_files) == 0:
			raise FileNotFoundError(f"No predictions found on {str(pred_dir)}")

		self.files = pred_files

	def load(self, img_file_name: str) -> dict:
		pred_file = self.root_dir / f"{img_file_name}.json"
		predictions = self._load_from_file(pred_file)

		return predictions

	def _load_from_file(self, pred_file):
		with pred_file.open('r') as f:
			predictions = json.load(f)

		for pred in predictions:
			pred['mask'] = mask_conversions.rle_to_bin_mask(pred['mask'])

		# Could test keys and values too, but whatever

		return predictions

	def copy_to(self, out_dir: Path):
		for file in self.files:
			shutil.copy(file, out_dir / file.name)