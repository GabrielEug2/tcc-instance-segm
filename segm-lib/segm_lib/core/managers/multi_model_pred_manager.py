from pathlib import Path


from segm_lib.core.managers import SingleModelPredManager
from segm_lib.core.classname_normalization import normalize_classname


class MultiModelPredManager:
	"""Functions to work with predictions from in segm_lib format.
	Each model predictions are kept in a separate folder, and should be
	manipulated using the SingleModelPredManager objects returned from
	get_manager().
	"""

	def __init__(self, pred_dir: Path):
		if not pred_dir.exists():
			pred_dir.mkdir(parents=True)

		self.root_dir = pred_dir

	def get_model_names(self) -> list[str]:
		return [f.stem for f in self.root_dir.glob('*') if f.is_dir()]	

	def get_manager(self, model_name: str) -> SingleModelPredManager:
		model_dir = self.root_dir / model_name
		return SingleModelPredManager(model_dir)
		
	def normalize_classnames(self):
		for model in self.get_model_names():
			manager = self.get_manager(model)

			for img in manager.get_img_names():
				predictions = manager.load(img)
				for pred in predictions:
					pred.classname = normalize_classname(pred.classname)
				manager.save(predictions, img)