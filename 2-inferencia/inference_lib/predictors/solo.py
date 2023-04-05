from pathlib import Path

from adet.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures import Boxes
import torch

from .base_pred import BasePred
from .config import config
from .format_utils import bin_mask_to_rle


class SoloPred(BasePred):
    def __init__(self):
        cfg = get_cfg()
        cfg.merge_from_file(str(Path(config['solo']['dir'], config['solo']['config_file'])))
        cfg.MODEL.WEIGHTS = str(Path(config['solo']['dir'], config['solo']['weights_file']))
        cfg.MODEL.SOLOV2.SCORE_THR = 0.5
        cfg.MODEL.DEVICE = 'cpu'

        self._model = DefaultPredictor(cfg)

    def predict(self, img):
        raw_predictions = self._inference(img)

        formatted_predictions = self._to_custom_format(raw_predictions)
        return formatted_predictions
    
    def _inference(self, img):
        raw_predictions = self._model(img)

        # Por algum motivo além da minha compreensão, o SOLO testa o score
        # de classificação ANTES de ter os scores "definitivos". Isso faz
        # com que ele retorne resultados com score abaixo do que foi solicitado.
        # Pra consertar isso:
        inds = (raw_predictions['instances'].scores > 0.5)
        raw_predictions['instances'] = raw_predictions['instances'][inds]

        return raw_predictions
        
    def _to_custom_format(self, raw_predictions):
        # Formato da saída do modelo:
        #   mesmo formato do Detectron, porque é baseado nele
        #   (see https://github.com/aim-uofa/AdelaiDet/blob/master/demo/predictor.py)
        #
        # Formato esperado:
        #   see inference_lib.predictors.base_pred

        formatted_predictions = []
        instances = raw_predictions['instances']

        # O SOLO prediz as máscaras direto, sem calcular bounding boxes.
        # Isso é legal, mas o problema é que o nome das classes no plot fica
        # tudo empilhado no canto. Felizmente eles tem um código pra
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

        for i in range(len(instances)):
            # Já está no intervalo certo, [0,N)
            class_id = instances.pred_classes[i].item()
            confidence = instances.scores[i].item()
            mask = instances.pred_masks[i]
            bbox = instances.pred_boxes.tensor[i].tolist()

            pred = {
                'class_id': class_id,
                'confidence': confidence,
                'mask': bin_mask_to_rle(mask),
                'bbox': bbox,
            }
            formatted_predictions.append(pred)

        return formatted_predictions