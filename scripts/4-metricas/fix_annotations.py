import argparse
from pathlib import Path
import json

MODEL_MAP_FILE = Path(__file__).parent / 'model_map.json'

def fix_annotations(ann_file: Path):
    # O problema aqui é que o dataset baixado usa outra numeração de classes,
    # então nós precisamos:
    #   * Dar as classes que já existem no COCO o mesmo ID, para que as
    #     predições dos modelos façam sentido
    #
    #   * Dar as classes que NÃO existem no COCO um ID único (ou talvez
    #     excluí-las? preciso testar se faz sentido manter elas)
    #
    #   * Atualizar as partes relevantes do arquivo, para fazer sentido
    #     com os novos ids

    with ann_file.open('r') as f:
        anns = json.load(f)

    # Primeiro eu só mapeio quais serão os novos IDs
    id_maps = compute_newids(anns)

    # Depois eu atualizo nas annotations, mantendo uma cópia das
    # anotações originais, caso precise
    backup_file = Path(str(ann_file) + ".bak")
    if not backup_file.exists():
        Path.rename(ann_file, backup_file)

    update_annotations(anns, id_maps)
    with ann_file.open('w') as f:
        json.dump(anns, f)

    return id_maps

def compute_newids(anns):
    name_newid_map = {}
    oldid_newid_map = {}

    # Classes do COCO continuam com o mesmo ID
    with MODEL_MAP_FILE.open('r') as f:
        model_map = json.load(f)

    for id_, name in model_map.items():
        name_newid_map[name] = int(id_)

    coco_class_list = model_map.values()

    # Classes do Openimages recebem...
    new_id_gen = _id_gen()
    for category in anns['categories']:
        name = category['name'].lower() # Openimages usa capitalized
        old_id = category['id']

        # TODO merge common classes (ex. person and man)
		# Move to annotations.py or just fix on annotations.py
		# # IDs que foram pulados (o "gap" que eu deixei entre os IDs do COCO
		# # e os novos IDs, no prepare_data.py) também precisam de um nome no
		# # metadata
		# ids = [int(x) for x in classmap.keys()]
		# skipped_ids = set(range(0, max(ids))) - set(ids)
		# for id_ in skipped_ids:
		#     classmap[str(id_)] = ''
		
        if name in coco_class_list:
            # o mesmo ID que eu já atribui, se existe no COCO
            new_id = name_newid_map[name]
        else:
            # um novo ID, se não existir
            new_id = next(new_id_gen)
            name_newid_map[name] = new_id
        
        oldid_newid_map[old_id] = new_id

    return { 'name_newid': name_newid_map, 'oldid_newid': oldid_newid_map }

def _id_gen():
    # Começa do 100, só pra deixar bem separado dos do COCO [0-80)
    next_id = 100
    while True:
        yield next_id
        next_id += 1

def update_annotations(anns, id_maps):
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

def save_newid_map(id_maps, out_dir):
    newid_name_map = { v: k for k, v in id_maps['name_newid'].items() }
    test_map_file = out_dir / "test_map.json"

    with test_map_file.open('w') as f:
        json.dump(newid_name_map, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ann_file', help='file containing the annotations')

    args = parser.parse_args()

    ann_file = Path(args.ann_file)
    id_maps = fix_annotations(ann_file)
    save_newid_map(id_maps, ann_file.parent)