
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from .base_pred import BasePred
from .config import config
from inference_lib.format_utils import bin_mask_to_rle


class MaskrcnnPred(BasePred):
    def __init__(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config['maskrcnn']['config_file']))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config['maskrcnn']['config_file'])
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = 'cpu'

        self._model = DefaultPredictor(cfg)

    def predict(self, img):
        raw_predictions = self._model(img)

        formatted_predictions = self._to_custom_format(raw_predictions)
        return formatted_predictions

    def _to_custom_format(self, raw_predictions):
        # Formato da saída do modelo:
        #   https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-format
        #
        # Formato esperado:
        #   see inference_lib.predictors.base_pred

        formatted_predictions = []

        instances = raw_predictions['instances']
        for i in range(len(instances)):
            class_id = instances.pred_classes[i].item() + 1 # Detectron faz de [0-N), no COCO é [1-N]
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