
import json

COCO_ANN_FILE = r"E:\Desktop\TCC\Dados\COCO\annotations\instances_val2017.json"
OUT_FILE = "coco_class_map.json"

with open(COCO_ANN_FILE) as f:
    anns = json.load(f)

class_map = {}
for category in anns["categories"]:
    class_map[category['name']] = category['id']

with open(OUT_FILE, 'w') as f:
    json.dump(class_map, f, indent=4)