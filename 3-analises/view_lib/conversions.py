from pathlib import Path
import json

from pycocotools import mask

COCO_CLASS_MAP_FILE = Path(__file__).parent / 'coco_class_map.json'

def fix_class_ids(ann_file):
    # O problema aqui é que o dataset baixado usa outra numeração de classes.
    # "Pessoa" não tem o id 1, algumas classes não existem no COCO... alguns
    # problemas que precisam ser consertados no arquivo de annotations antes
    # de usá-lo para avaliação.
    with Path(ann_file).open('r') as f:
        anns = json.load(f)

    with COCO_CLASS_MAP_FILE.open('r') as f:
        coco_class_map = json.load(f)

    # Primeiro eu só mapeio qual serão os IDs novos
    classname_newid_map = {}
    oldid_newid_map = {}

    for category in anns['categories']:
        category_name = category['name'].lower() # Openimages usa capitalized

        if category_name in coco_class_map:
            # Dá o número certo, baseado no mapeamento do COCO
            new_id = coco_class_map[category_name]
        else:
            # É uma classe exclusiva do OpenImages, não pode ter ID conflitante
            # com os do COCO
            new_id = _gen_new_id()
        
        classname_newid_map[category_name] = new_id
        oldid_newid_map[category['id']] = new_id

    # Depois eu atualizo nas annotations
    for category in anns['categories']:
        category['id'] = classname_newid_map[category['name'].lower()]
    for ann in anns['annotations']:
        ann['category_id'] = oldid_newid_map[ann['category_id']]

    # Salva em um arquivo separado, caso queira ver as anotações originais
    new_ann_file = Path(ann_file).parent / (ann_file.stem + '_fix.json')
    with new_ann_file.open('w') as f:
        json.dump(anns, f)

def _gen_new_id():
    # Começa do 100, pra deixar separado dos do COCO (1-80)
    new_id = 100
    while True:
        yield new_id
        new_id += 1

def fix_img_ids(pred_files, ann_file):
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

        # Salva em um arquivo separado, caso queira ver as predições originais
        fixed_pred_file = pred_file.parent / (pred_file.stem + '_fix.json')
        with fixed_pred_file.open('w') as f:
            json.dump(predictions, f)

# def rle_to_bin(rle):
#     rle['counts'] = rle['counts'].encode('ascii')
#     bin_mask = mask.decode(rle)

#     return bin_mask