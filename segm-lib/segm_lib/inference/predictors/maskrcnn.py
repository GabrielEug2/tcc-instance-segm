
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances

from .abstract_predictor import Predictor
from .prediction import Prediction
from .config import config


class Maskrcnn(Predictor):
	def __init__(self):
		cfg = get_cfg()
		cfg.merge_from_file(model_zoo.get_config_file(config['maskrcnn']['config_file']))
		cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config['maskrcnn']['config_file'])
		cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
		cfg.MODEL.DEVICE = 'cpu'

		self._model = DefaultPredictor(cfg)

	def predict(self, img) -> list[Prediction]:
		instances = self._model(img)['instances']

		formatted_predictions = self._to_custom_format(instances)
		return formatted_predictions

	def _to_custom_format(self, instances: Instances):
		# Formato da saída do modelo:
		#   https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-format
		#
		# Formato esperado:
		#   see inference_lib.predictors.base_pred

		formatted_predictions = []
		for i in range(len(instances)):
			class_id = instances.pred_classes[i].item()
			classname = self._id_to_name(class_id)

			confidence = instances.scores[i].item()
			
			mask = instances.pred_masks[i]

			x1, y1, x2, y2 = instances.pred_boxes.tensor[i].tolist()
			w = x2 - x1
			h = y2 - y1
			bbox = [x1, y1, w, h]

			formatted_predictions.append(Prediction(classname, confidence, mask, bbox))

		return formatted_predictions
	
	def _id_to_name(self, class_id):
		# Cada modelo tem sua próprio mapeamento de ID pra nome, dependendo de
		# onde ele foi treinado. No meu caso, os três modelos seguem a mesma
		# numeração, a do COCO, mas achei importante deixar cada modelo com o
		# sua própria função de conversão
		return super().cocoid_to_classname(class_id)