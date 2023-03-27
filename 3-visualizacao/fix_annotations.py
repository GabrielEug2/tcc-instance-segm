from pathlib import Path
import json

COCO_CLASS_MAP_FILE = Path(__file__).parent / 'coco_class_map.json'

def fix_annotations(ann_file):
    # O problema aqui é que o dataset baixado usa outra numeração de classes,
    # então nós precisamos:
    #   * Dar as classes que já existem no COCO o mesmo ID, para que as
    #     predições os modelos façam sentido
    #
    #   * Dar as classes que NÃO existem no COCO um ID único (ou talvez excluí-las?
    #     preciso testar se faz sentido manter elas)
    #
    #   * Adicionar as classes do COCO, para que a API consiga plotar
    #     as detecções com o nome da classe
    #
    with Path(ann_file).open('r') as f:
        anns = json.load(f)
    with COCO_CLASS_MAP_FILE.open('r') as f:
        coco_class_map = json.load(f)

    # Primeiro eu só mapeio qual serão os IDs novos
    class_newid_map = {}
    oldid_newid_map = {}
    new_id_gen = _id_gen()

    # Classes do COCO continuam com o mesmo ID
    for name, id in coco_class_map.items():
        class_newid_map[name] = id

    for category in anns['categories']:
        category_name = category['name'].lower() # Openimages usa capitalized
        old_id = category['id']

        if category_name in class_newid_map:
            new_id = class_newid_map[category_name]
        else:
            new_id = next(new_id_gen)
            class_newid_map[category_name] = new_id
        
        oldid_newid_map[old_id] = new_id

    # Depois eu atualizo nas annotations
    categories = []
    for name, id in class_newid_map.items():
        categories.append({
            'name': name,
            'id': id,
            'supercategory': None
        })
    anns['categories'] = categories

    for ann in anns['annotations']:
        ann['category_id'] = oldid_newid_map[ann['category_id']]

    new_ann_file = Path(ann_file).parent / (ann_file.stem + '_fix.json')
    with new_ann_file.open('w') as f:
        json.dump(anns, f)

def _id_gen():
    # Começa do 100, pra deixar separado dos do COCO (1-80)
    next_id = 100
    while True:
        yield next_id
        next_id += 1