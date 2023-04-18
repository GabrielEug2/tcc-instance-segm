MODEL_MAP = {
	'maskrcnn': None,
	'yolact': None,
	'solo': None,
}

def load_models(models):
	for model_name in models:
		try:
			import_model(model_name)
		except ImportError as e:
			raise ImportError(f"\"{model_name}\" is implemented on the library, but not installed properly.") from e
		except ValueError:
			raise ValueError(f"\"{model_name}\" is not implemented. How do you expect to run that.")

def import_model(model_name):
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