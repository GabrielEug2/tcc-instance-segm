from pathlib import Path
import sys

import torch

from .base_predictor import Predictor
from .config import config

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
		raw_predictions = postprocess(preds, w, h, score_threshold = 0.5)
	
		formatted_predictions = self._to_custom_format(raw_predictions)
		return formatted_predictions

	def _to_custom_format(self, raw_predictions):
		# Formato da saída do modelo:
		#   https://github.com/dbolya/yolact/blob/master/layers/output_utils.py
		#
		# Formato esperado:
		#   see inference_lib.predictors.base_pred

		formatted_predictions = []

		for i in range(len(raw_predictions[0])):
			# Pelo que eu entendi Yolact faz de [0-N], com 0 sendo o background,
			# então nós precisamos consertar isso subtraindo -1
			# EDIT aparentemente não, já que fazer isso faz com que fiquem
			#      redictions com classe "-1"
			# TODO confirmar isso no plot (pessoa tem que estar certo)
			class_id = raw_predictions[0][i].item()
			classname = self._id_to_name(class_id)
			confidence = raw_predictions[1][i].item()
			mask = raw_predictions[3][i]
			bbox = raw_predictions[2][i].tolist()

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
		return super.cocoid_to_classname(class_id)