import json
from pathlib import Path

from segm_lib import mask_conversions

class PredManager:
	def __init__(self, root_dir: Path):
		self.root_dir = root_dir
		
	def save(self, preds: list, img_file_name: str, model_name: str):
		"""Save the predictions for a given model on a given image.

		Args:
			preds (list): list of predictions, as described in
				inference.predictors.abstract_predictor.
			img_file_name (str): image the predictions refer to.
			model_name (str): name of the model used to make the
				predictions.
		"""
		for pred in preds:
			pred['mask'] = mask_conversions.bin_mask_to_rle(pred['mask'])

		out_file = self.root_dir / model_name / f"{img_file_name}.json"
		out_file.parent.mkdir(parents=True, exist_ok=True)
		with out_file.open('w') as f:
			json.dump(preds, f)