from pathlib import Path

from adet.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures import Boxes, Instances
import torch

from .abstract_predictor import Predictor
from .config import config

class Solo(Predictor):
	def __init__(self):
		cfg = get_cfg()
		cfg.merge_from_file(str(Path(config['solo']['dir'], config['solo']['config_file'])))
		cfg.MODEL.WEIGHTS = str(Path(config['solo']['dir'], config['solo']['weights_file']))
		cfg.MODEL.SOLOV2.SCORE_THR = 0.5
		cfg.MODEL.DEVICE = 'cpu'
		cfg.SOLVER.IMS_PER_BATCH = 1

		self._model = DefaultPredictor(cfg)

	def predict(self, img):
		instances = self._model(img)['instances']
		# Por algum motivo além da minha compreensão, o SOLO testa o score
		# de classificação ANTES de ter os scores "definitivos". Isso faz
		# com que ele retorne resultados com score abaixo do que foi solicitado.
		# Pra consertar isso:
		inds = (instances.scores > 0.5)
		instances = instances[inds]

		formatted_predictions = self._to_custom_format(instances)
		return formatted_predictions
	
	def _to_custom_format(self, instances: Instances):
		# Formato da saída do modelo:
		#   mesmo formato do Detectron, porque é baseado nele
		#   (see https://github.com/aim-uofa/AdelaiDet/blob/master/demo/predictor.py)
		#
		# Formato esperado:
		#   see inference_lib.predictors.base_pred

		# O SOLO prediz as máscaras direto, sem calcular bounding boxes.
		# Isso é legal, mas o problema é que, na hora de plotar, as APIs
		# normalmente plotam o nome da classe no canto da bounding box.
		# Então eu preciso delas. Felizmente eles tem um código pra
		# computar as bounding boxes a partir das máscaras.
		#
		# See:
		#   https://github.com/aim-uofa/AdelaiDet/issues/469
		#   https://github.com/aim-uofa/AdelaiDet/blob/4a3a1f7372c35b48ebf5f6adc59f135a0fa28d60/adet/modeling/solov2/solov2.py#L496
		pred_boxes = torch.zeros(instances.pred_masks.size(0), 4)
		for i in range(instances.pred_masks.size(0)):
			mask = instances.pred_masks[i].squeeze()
			ys, xs = torch.where(mask)
			pred_boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()]).int()
		instances.pred_boxes = Boxes(pred_boxes)

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

			pred = {
				'classname': classname,
				'confidence': confidence,
				'mask': mask,
				'bbox': bbox,
			}
			formatted_predictions.append(pred)

		return formatted_predictions
	
	def _id_to_name(self, class_id):
		# Cada modelo tem sua próprio mapeamento de ID pra nome, dependendo de
		# onde ele foi treinado. No meu caso, os três modelos seguem a mesma
		# numeração, a do COCO, mas achei importante deixar cada modelo com o
		# sua própria função de conversão
		return super().cocoid_to_classname(class_id)