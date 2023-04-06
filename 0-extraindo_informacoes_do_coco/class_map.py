
import json
from pathlib import Path

# Não importa qual dos dois (train ou val), eles tem as mesmas classes.
# Eu só uso o val porque ele é menor / carrega mais rápido.
COCO_ANN_FILE = "/mnt/e/Desktop/TCC/Dados/COCO/annotations/instances_val2017.json"

DEFAULT_MAP_FILE = Path(__file__).parent / "default_map.json"
MODEL_MAP_FILE = Path(__file__).parent / "model_map.json"

with open(COCO_ANN_FILE) as f:
    anns = json.load(f)

# O mapeamento padrão vai de 1 a 90, pulando alguns números,
# o que é confuso por si só e irritante de lidar
default_coco_map = {}
for category in anns["categories"]:
    default_coco_map[category['id']] = category['name']

with DEFAULT_MAP_FILE.open('w') as f:
    json.dump(default_coco_map, f, indent=4)

# A numeração que os modelos usam, e que nós usaremos aqui,
# é de 0 a N (80), sequencial, seguindo a ordem do COCO
model_map = {}
sorted_categories = sorted(default_coco_map.items(), key=lambda c: c[0])
n_classes = len(default_coco_map.keys())
for i in range(n_classes):
    model_map[i] = sorted_categories[i][1]

with MODEL_MAP_FILE.open('w') as f:
    json.dump(model_map, f, indent=4)