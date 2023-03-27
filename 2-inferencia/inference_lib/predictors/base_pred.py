
from abc import ABC, abstractmethod

class BasePred(ABC):
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
                {
                    'class_id': int,
                    'confidence': float,
                    'mask': RLE,
                    'bbox': [x1, y1, x2, y2],
                }
        """
        pass