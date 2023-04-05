import argparse
from pathlib import Path
import json


MODEL_MAP_FILE = Path(__file__).parent / 'model_map.json'


def fix_annotations(anns):
    # O problema aqui é que o dataset baixado usa outra numeração de classes,
    # então nós precisamos:
    #   * Dar as classes que já existem no COCO o mesmo ID, para que as
    #     predições dos modelos façam sentido
    #
    #   * Dar as classes que NÃO existem no COCO um ID único (ou talvez excluí-las?
    #     preciso testar se faz sentido manter elas)
    #
    #   * Atualizar o campo "categories" para fazer sentido com os novos ids

    # Primeiro eu só mapeio quais serão os novos IDs
    id_maps = _compute_ids(anns)

    # Depois eu atualizo nas annotations, mantendo uma cópia das
    # anotações originais, caso precise
    backup_file = Path(str(ann_file) + ".bak")
    if not backup_file.exists():
        Path.rename(ann_file, backup_file)

    _update_annotations(anns, id_maps)
    with ann_file.open('w') as f:
        json.dump(anns, f)

    return id_maps

def _compute_ids(anns):
    name_newid_map = {}
    oldid_newid_map = {}
    new_id_gen = _id_gen()

    with MODEL_MAP_FILE.open('r') as f:
        model_map = json.load(f)
    coco_class_list = model_map.values()

    # Classes do COCO continuam com o mesmo ID
    for id_, name in model_map.items():
        name_newid_map[name] = int(id_)

    # Classes do Openimages recebem...
    for category in anns['categories']:
        name = category['name'].lower() # Openimages usa capitalized
        old_id = category['id']

        if name in coco_class_list:
            # o mesmo ID que os modelos usam, se existe no COCO
            new_id = name_newid_map[name]
        else:
            # um novo ID, se não existir
            new_id = next(new_id_gen)
            name_newid_map[name] = new_id
        
        oldid_newid_map[old_id] = new_id

    return { 'name_newid': name_newid_map, 'oldid_newid': oldid_newid_map }

def _id_gen():
    # Começa do 100, pra deixar separado dos do COCO (1-90)
    next_id = 100
    while True:
        yield next_id
        next_id += 1

def _update_annotations(anns, id_maps):
    for ann in anns['annotations']:
        ann['category_id'] = id_maps['oldid_newid'][ann['category_id']]

    categories = []
    sorted_name_id = sorted(id_maps['name_newid'].items(), key=lambda c: c[1])
    for name, id_ in sorted_name_id:
        categories.append({
            'name': name,
            'id': id_,
            'supercategory': None
        })
    anns['categories'] = categories


parser = argparse.ArgumentParser()
parser.add_argument('ann_file', help='file containing the annotations')

args = parser.parse_args()

ann_file = Path(args.ann_file)
with ann_file.open('r') as f:
    anns = json.load(f)

id_maps = fix_annotations(anns)

newid_name_map = { v: k for k, v in id_maps['name_newid'].items() }
test_map_file = ann_file.parent / "classmap.json"
with test_map_file.open('w') as f:
    json.dump(newid_name_map, f, indent=4)