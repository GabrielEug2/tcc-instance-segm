
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

	# def __iter__(self):
	# 	self._current_index = 0
	# 	return self
	# def __next__(self):
	# 	if self._current_index >= len(self._preds):
	# 		raise StopIteration

	# 	pred = self._preds[self._current_index]
	# 	self._current_index += 1

	# 	return pred

	# def get_n_objects(self):
	# 	return len(self._preds)