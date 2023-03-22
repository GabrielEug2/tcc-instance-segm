
import json

ANN_FILES = [
    r"E:\Desktop\TCC\Datasets\COCO\annotations\instances_train2017.json",
    r"E:\Desktop\TCC\Datasets\COCO\annotations\instances_val2017.json"
]
OUT_FILE = "class_dist_coco.json"

class_dist = {}
for file in ANN_FILES:
    with open(file) as f:
        anns = json.load(f)

    for ann in anns["annotations"]:
        class_dist[ann["category_id"]] = class_dist.get(ann["category_id"], 0) + 1

# Eu quero o nome, não o id
for category in anns["categories"]:
    n_occurences = class_dist.pop(category["id"])
    class_dist[category["name"]] = n_occurences

class_dist = dict(sorted(class_dist.items(), key=lambda item: item[1], reverse=True))

with open(OUT_FILE, 'w') as f:
    json.dump(class_dist, f, indent=4)