from typing import Union
from .abstract_predictor import Predictor as _Predictor

VALID_MODELS = ['maskrcnn', 'yolact', 'solo']
MODEL_MAP: dict[str, _Predictor] = {}

def import_model(model_name):
	# Imports are conditional because you don't need
	# to install all the models
	match model_name:
		case 'maskrcnn':
			from .maskrcnn import Maskrcnn
			model = Maskrcnn
		case 'yolact':
			from .yolact import Yolact
			model = Yolact
		case 'solo':
			from .solo import Solo
			model = Solo
		case other:
			raise ValueError()
		
	MODEL_MAP[model_name] = model