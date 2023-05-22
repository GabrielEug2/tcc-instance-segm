from pathlib import Path
import json

# In my case, all models were trained on COCO, so the
# class output range is the same for all three
# If the models were trained somewhere else, that wouldn't be true
with Path('funcionalidades/0-informacoes_do_coco/coco_classmap_default.json').open('r') as f:
    coco_classmap = json.load(f)
coco_classes = list(coco_classmap.keys())

possible_classes_dir = Path('possible_classes')
possible_classes_dir.mkdir(exist_ok=True)
model_names = ['maskrcnn', 'yolact', 'solo']
for model in model_names:
    out_file = possible_classes_dir / f'{model}.json'
   
    with out_file.open('w') as f:
    	json.dump(coco_classes, f)