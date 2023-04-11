from .maskrcnn import Maskrcnn
from .solo import Solo
from .yolact import Yolact

MODEL_MAP = {
    'maskrcnn': Maskrcnn(),
    'yolact': Yolact(),
    'solo': Solo(),
}