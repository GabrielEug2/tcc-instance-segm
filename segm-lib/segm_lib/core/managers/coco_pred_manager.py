
import json
from pathlib import Path


class COCOPredManager:
	def __init__(self, pred_file: Path):
		if not pred_file.exists():
			pred_file.parent.mkdir(parents=True, exist_ok=True)

			self.file = pred_file
			self.predictions = []
			self.save()
			return

		with pred_file.open('r') as f:
			preds = json.load(f)
		# Could test keys and values too, but whatever

		self.file = pred_file
		self.predictions = preds

	def save(self):
		with self.file.open('w') as f:
			json.dump(self.predictions, f)