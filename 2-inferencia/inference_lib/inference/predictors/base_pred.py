
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
                    "class_id": int, seguindo a ordem do COCO (pessoa, bicicleta,
                        carro...), mas mapeado para o intervalo [0-80). See
                        "2-inferencia/class_map.json"
                    "confidence": float, entre 0 e 1, com 1 sendo 100% de certeza,
                    "mask": compact RLE, que é a forma que o COCO usa para 
                        comprimir máscaras,
                    "bbox": [x1, y1, x2, y2],
                }
        """
        pass