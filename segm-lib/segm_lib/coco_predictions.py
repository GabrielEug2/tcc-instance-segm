
import json
from pathlib import Path

class COCOPredictions:
	def __init__(self, pred_file: Path = None):
		self.file = pred_file

	def save(self, coco_preds):
		self.file.parent.mkdir(parents=True, exist_ok=True)
		with self.file.open('w') as f:
			json.dump(coco_preds, f)
	
	@classmethod
	def from_file(out_file):
		pass
# 		with pred_file.open('r') as f:
# 			preds = json.load(f)

# 		# Could test keys and values too, but whatever

# 		self._preds = preds

# 	def get_n_objects(self):
# 		return len(self._preds)