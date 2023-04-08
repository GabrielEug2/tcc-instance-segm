## Extraindo informações do COCO

Você não precisa necessariamente rodar nada dessa pasta, já que os arquivos resultantes já estão inclusos no repositório, mas se quiser rodar...

Requisitos:
* Anotações do COCO ([train/val 2017](https://cocodataset.org/#download)) - não precisa das imagens.

Como usar:
* Para calcular a distribuição de classes:
    * `python class_dist.py <ann_dir>`

* Para gerar uma legenda das classes (pares de ID-nome da classe, para facilitar a interpretação):
    * `python class_map.py <ann_dir>`