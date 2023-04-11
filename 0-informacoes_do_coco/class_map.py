import argparse
import json
from pathlib import Path

import stats_lib

parser = argparse.ArgumentParser()
parser.add_argument('ann_dir', help='Directory where you placed the annotations')

args = parser.parse_args()

# Não importa qual arquivo (train ou val), eles tem as mesmas classes.
# Eu só uso o val porque ele é menor / carrega mais rápido.
ann_file = Path(args.ann_dir, 'instances_val2017.json')
classmaps = stats_lib.compute_classmap(ann_file)

for map_name in classmaps:
	map_file = Path(__file__).parent / f"classmap_{map_name}.json"
	with map_file.open('w') as f:
		json.dump(classmaps[map_name], f, indent=4)