
## Misc

Scripts aleatórios que eu usei durante o desenvolvimento.

* `class_dist_coco.py`: calcula a distribuição de classes nos objetos do COCO.
    * Requisitos:
        * Anotações do COCO ([train/val 2017](https://cocodataset.org/#download))

    * Como usar:
        * Edite no arquivo o caminho das annotations.
        * `python class_dist_coco.py`

* `download_test_set.py`: baixa imagens e anotações do OpenImages, com algumas das classes que tem mais objetos no COCO.
    * Requisitos:
        * [FiftyOne](https://docs.voxel51.com/getting_started/install.html)

    * Como usar:
        * `python download_test_set.py <n_imgs> <out_dir>`