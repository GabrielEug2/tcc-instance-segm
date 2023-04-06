## Extraindo informações do COCO

Você não precisa necessariamente rodar nada dessa pasta, já que os arquivos resultantes já estão inclusos no repositório, mas se quiser rodar...

Requisitos:
* Anotações do COCO ([train/val 2017](https://cocodataset.org/#download)) - não precisa das imagens.

Como usar:
* `class_dist.py`: calcula a distribuição de classes no COCO.
    * `python class_dist.py <ann_dir>`

* `class_map.py`: gera uma legenda das classes (pares de ID-nome da classe, para facilitar a interpretação).
    * `python class_map.py <ann_dir>`