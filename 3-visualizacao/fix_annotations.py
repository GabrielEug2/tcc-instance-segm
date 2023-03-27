import argparse
from pathlib import Path
import json


COCO_CLASS_MAP_FILE = Path(__file__).parent / 'coco_class_map.json'


def fix_annotations(ann_file_str):
    # O problema aqui é que o dataset baixado usa outra numeração de classes,
    # então nós precisamos:
    #   * Dar as classes que já existem no COCO o mesmo ID, para que as
    #     predições os modelos façam sentido
    #
    #   * Dar as classes que NÃO existem no COCO um ID único (ou talvez excluí-las?
    #     preciso testar se faz sentido manter elas)

    # Primeiro eu só mapeio quais serão os novos IDs
    name_newid_map = {}
    oldid_newid_map = {}
    new_id_gen = _id_gen()

    with COCO_CLASS_MAP_FILE.open('r') as f:
        coco_class_map = json.load(f)

    for name, coco_id in coco_class_map.items():
        # Classes do COCO continuam com o mesmo ID
        name_newid_map[name] = coco_id

    ann_file = Path(ann_file_str)
    with ann_file.open('r') as f:
        anns = json.load(f)

    for category in anns['categories']:
        category_name = category['name'].lower() # Openimages usa capitalized
        old_id = category['id']

        # Classes do Openimages recebem...
        if category_name in name_newid_map:
            # o ID do COCO, se existir no COCO
            new_id = name_newid_map[category_name]
        else:
            # um novo ID, se não existir
            new_id = next(new_id_gen)
            name_newid_map[category_name] = new_id
        
        oldid_newid_map[old_id] = new_id

    # Depois eu atualizo nas annotations
    categories = []
    for name, id in name_newid_map.items():
        categories.append({
            'name': name,
            'id': id,
            'supercategory': None
        })
    anns['categories'] = categories

    for ann in anns['annotations']:
        ann['category_id'] = oldid_newid_map[ann['category_id']]

    Path.rename(ann_file, ann_file.parent / (ann_file.stem + "_old.json"))
    new_ann_file = Path(ann_file_str)
    with new_ann_file.open('w') as f:
        json.dump(anns, f)

def _id_gen():
    # Começa do 100, pra deixar separado dos do COCO (1-90?)
    next_id = 100
    while True:
        yield next_id
        next_id += 1

parser = argparse.ArgumentParser()
parser.add_argument('ann_file', help='file containing the annotations')

args = parser.parse_args()

fix_annotations(args.ann_file)