from pathlib import Path
import time

import cv2
import torch
from adet.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures import Boxes

from .predictor import Predictor
from .config import config
from inference_lib.format_utils import bin_mask_to_rle

class SoloPred(Predictor):
    def __init__(self):
        cfg = get_cfg()
        cfg.merge_from_file(str(Path(config['solo']['dir'], config['solo']['config_file'])))
        cfg.MODEL.WEIGHTS = str(Path(config['solo']['dir'], config['solo']['weights_file']))
        cfg.MODEL.SOLOV2.SCORE_THR = 0.5
        cfg.MODEL.DEVICE = 'cpu'

        self._model = DefaultPredictor(cfg)

    def predict(self, img_path):
        img = cv2.imread(str(img_path))

        start_time = time.time()
        raw_predictions = self._model(img)
        inference_time = time.time() - start_time

        # Por algum motivo além da minha compreensão, o SOLO testa o score
        # de classificação ANTES de ter os scores "definitivos". Isso faz
        # com que ele retorne resultados com score abaixo do que foi solicitado.
        # Pra consertar isso:
        inds = (raw_predictions['instances'].scores > 0.5)
        raw_predictions['instances'] = raw_predictions['instances'][inds]

        # Formato da saída do modelo: mesmo formato do Detectron, porque é
        # baseado nele (see https://github.com/aim-uofa/AdelaiDet/blob/master/demo/predictor.py)
        predictions = self._to_dict(raw_predictions)

        return predictions, inference_time
    
    def _to_dict(self, raw_predictions):
        # O SOLO prediz as máscaras direto, sem calcular bounding boxes.
        # O único problema disso é que o nome das classes no plot fica
        # tudo empilhado no canto. Felizmente eles tem um código pra
        # computar as bounding boxes a partir das máscaras.
        pred_boxes = torch.zeros(raw_predictions.pred_masks.size(0), 4)
        for i in range(raw_predictions.pred_masks.size(0)):
            mask = raw_predictions.pred_masks[i].squeeze()
            ys, xs = torch.where(mask)
            pred_boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()]).int()
        raw_predictions.pred_boxes = Boxes(pred_boxes)


        formatted_predictions = []

        instances = raw_predictions['instances']
        for i in range(len(instances)):
            pred_class = instances.pred_classes[i].item()
            score = instances.scores[i].item()
            mask = instances.pred_masks[i].tolist()
            bbox = instances.pred_boxes.tensor[i].tolist()

            pred = {
                'class_id': pred_class,
                'score': score,
                'mask': bin_mask_to_rle(mask),
                'bbox': bbox,
            }
            formatted_predictions.append(pred)

        return formatted_predictions