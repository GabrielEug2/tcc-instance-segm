from pathlib import Path
import json

from pycocotools import mask

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

def fix_predictions(pred_files, ann_file):
    # No momento de inferência, o ID da imagem não importa, eu só salvo como
    # o nome do arquivo pra ficar fácil de achar. Aqui, no entanto, eu preciso
    # que o ID seja o mesmo que está no arquivo de annotations.
    with ann_file.open('r') as f:
        anns = json.load(f)

    filename_id_map = {}
    for img in anns['images']:
        filename_id_map[img['file_name']] = img['id']

    for pred_file in pred_files:
        with pred_file.open('r') as f:
            predictions = json.load(f)

        for prediction in predictions:
            filename = prediction['image_id'] + '.jpg'
            prediction['image_id'] = filename_id_map[filename]

            # Convert to [top-left-x, top-left-y, width, height]
            # in relative coordinates in [0, 1] x [0, 1]
            x1, y1, x2, y2 = box
            rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

            # Também adiciona um bbox porque o fiftyone precisa disso pra importar
            prediction['bbox'] = [0, 0, 0, 0]

        fixed_pred_file = pred_file.parent / (pred_file.stem + '_fix.json')
        with fixed_pred_file.open('w') as f:
            json.dump(predictions, f)

def rle_to_bin(rle):
    rle['counts'] = rle['counts'].encode('ascii')
    bin_mask = mask.decode(rle)

    return bin_mask