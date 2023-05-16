from pathlib import Path
import sys

import torch

from .abstract_predictor import Predictor
from .config import config
from segm_lib.pred_manager import Prediction

sys.path.insert(0, config['yolact']['dir'])
from data import set_cfg
from data import cfg
from yolact import Yolact as YolactLib
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess


class Yolact(Predictor):
	# Based on:
	# https://github.com/dbolya/yolact/issues/256#issuecomment-567371328

	def __init__(self):
		set_cfg(config['yolact']['config_name'])
		cfg.mask_proto_debug = False

		model = YolactLib()
		model.load_weights(str(Path(config['yolact']['dir'], config['yolact']['weights_file'])))
		model.eval()

		self._model = model
		
	def predict(self, img):
		with torch.no_grad():
			frame = torch.from_numpy(img).float()
			batch = FastBaseTransform()(frame.unsqueeze(0))
			preds = self._model(batch)
			# Essas predições ainda não são finais, falta converter
			# as máscaras pro formato certo.
		h, w, _ = img.shape
		classes, scores, boxes, masks = postprocess(preds, w, h, score_threshold = 0.5)
	
		formatted_predictions = self._to_custom_format(classes, scores, boxes, masks)
		return formatted_predictions

	def _to_custom_format(self, classes, scores, boxes, masks):
		# Formato da saída do modelo:
		#   https://github.com/dbolya/yolact/blob/master/layers/output_utils.py
		#
		# Formato esperado:
		#   see inference_lib.predictors.base_pred

		formatted_predictions = []
		for i in range(len(masks)):
			class_id = classes[i].item()
			classname = self._id_to_name(class_id)

			confidence = scores[i].item()

			mask = masks[i].to(torch.bool)

			x1, y1, x2, y2 = boxes[i].tolist()
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