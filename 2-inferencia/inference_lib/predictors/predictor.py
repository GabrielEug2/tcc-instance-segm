
from abc import ABC, abstractmethod

class Predictor(ABC):
    def __init__(self):
        """Inicializa o modelo, carregando pesos e configurando o
        que for necessário."""
        pass
    
    @abstractmethod
    def predict(self, img_path):
        """Segmenta objetos na imagem.

        Args:
            img_path (str): caminho para a imagem

        Returns:
            list: a list of detections for the image, in the format:
            {
                'class_id': int,
                'confidence': float,
                'mask': RLE,
                'bbox': [x1, y1, x2, y2],
            }
            inference_time: time it took to run inference, excluding the
                formatting bits (plotting on the img, etc)
        """
        pass