## Extraindo informações do COCO

Você não precisa necessariamente rodar nada dessa pasta, já que os arquivos resultantes já estão inclusos no repositório, mas se quiser rodar...

Requisitos:
* Anotações do COCO ([train/val 2017](https://cocodataset.org/#download)) - não precisa das imagens.
* Stats lib
	```bash
	pip install -e ../personal_lib/stats
	```

Como usar:
* `python class_dist.py <ann_dir>`
* `python class_map.py <ann_dir>`