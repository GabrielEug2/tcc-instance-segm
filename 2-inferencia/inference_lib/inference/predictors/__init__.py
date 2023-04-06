from .maskrcnn import MaskrcnnPred
from .solo import SoloPred
from .yolact import YolactPred

MODEL_MAP = {
    'maskrcnn': MaskrcnnPred(),
    'yolact': YolactPred(),
    'solo': SoloPred(),
}