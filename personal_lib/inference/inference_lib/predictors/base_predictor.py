
from pathlib import Path
import json
from abc import ABC, abstractmethod

COCO_CLASSMAP_FILE = Path(__file__).parent / 'coco_classmap_model.json'
with COCO_CLASSMAP_FILE.open('r') as f:
	COCO_CLASSMAP = json.load(f)

class Predictor(ABC):
	def __init__(self):
		"""Inicializa o modelo, carregando pesos e configurando o
		que for necessário."""
		pass
	
	@abstractmethod
	def predict(self, img):
		"""Segmenta objetos na imagem.

		Args:
			img (np.ndarray): imagem no formato BGR

		Returns:
			list: lista de objetos detectados na imagem, no formato:
				{"classname": string,
				"confidence": float, entre 0 e 1, com 1 sendo 100% de certeza,
				"mask": torch.Tensor, de tipo bool
				"bbox": [x1, y1, x2, y2]}
		"""
		pass
	
	@classmethod
	def cocoid_to_classname(cls, id):
		return COCO_CLASSMAP[str(id)]