from . import maskrcnn, yolact, solo

MODELS = [
    {'name': 'maskrcnn', 'module': maskrcnn},
    {'name': 'yolact', 'module': yolact},
    {'name': 'solo', 'module': solo}
]