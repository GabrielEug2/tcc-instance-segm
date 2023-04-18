import argparse
import json
from pathlib import Path

from personal_lib import ann_parser

parser = argparse.ArgumentParser()
parser.add_argument('ann_dir', help='Directory where you placed the annotations')
args = parser.parse_args()

# Não importa qual arquivo (train ou val), os dois tem as mesmas classes.
# Eu só uso o val porque ele é menor / carrega mais rápido.
ann_file = Path(args.ann_dir, 'instances_val2017.json')
anns = ann_parser.load_anns(ann_file)

# O COCO pula alguns IDs: tem 80 classes, mas vai até o ID 90.
# Para simplificar, os modelos normalizam pra [0,N)
default_coco_map = ann_parser.raw_classmap(anns)
model_map = ann_parser.normalized_classmap(default_coco_map)
classmaps = { 'default': default_coco_map, 'model': model_map }

for map_name in classmaps:
	map_file = Path(__file__).parent / f"coco_classmap_{map_name}.json"
	with map_file.open('w') as f:
		json.dump(classmaps[map_name], f, indent=4)