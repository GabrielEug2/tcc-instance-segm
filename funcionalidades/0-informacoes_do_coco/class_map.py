import argparse
import json
from pathlib import Path

from segm_lib.coco_annotations import COCOAnnotations

parser = argparse.ArgumentParser()
parser.add_argument('ann_dir', help='Directory where you placed the annotations')
args = parser.parse_args()

# Não importa qual arquivo (train ou val), os dois tem as mesmas classes.
# Eu só uso o val porque ele é menor / carrega mais rápido.
ann_file = Path(args.ann_dir, 'instances_val2017.json')

# O COCO pula alguns IDs: tem 80 classes, mas vai até o ID ~90.
# Para simplificar, os modelos normalizam pra [0,N)
coco_anns = COCOAnnotations.from_file(ann_file)
default_map = coco_anns.classmap()
normalized_map = coco_anns.normalized_classmap()
classmaps = { 'default': default_map, 'normalized': normalized_map }

for map_name in classmaps:
	map_file = Path(__file__).parent / f"coco_classmap_{map_name}.json"
	with map_file.open('w') as f:
		json.dump(classmaps[map_name], f, indent=4)