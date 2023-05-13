
import json
from pathlib import Path

class COCOPredictions:
	def __init__(self, pred_file: Path):
		with pred_file.open('r') as f:
			preds = json.load(f)

		# Could test keys and values too, but whatever

		self._preds = preds

	def get_n_objects(self):
		return len(self._preds)