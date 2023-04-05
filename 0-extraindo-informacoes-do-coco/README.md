## Extraindo informações do COCO

Você não precisa necessariamente rodar nada dessa pasta, os arquivos resultantes já estão incluídos no repositório.

Requisitos:
* Anotações do COCO ([train/val 2017](https://cocodataset.org/#download)) (não precisa das imagens)

Como usar:
* `class_dist_coco.py`: calcula a distribuição de classes nos objetos do COCO.
    * Edite no arquivo o caminho das annotations.
    * `python class_dist_coco.py`

* `class_class_map.py`: calcula a distribuição de classes nos objetos do COCO.
    * Edite no arquivo o caminho das annotations.
    * `python class_dist_coco.py`