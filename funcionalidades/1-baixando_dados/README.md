## Baixando dados

Para esse projeto, eu usei dados do Openimages, com algumas das classes que mais aparecem no COCO. Você pode baixar uma dataset com essas mesmas características seguindo os passos abaixo.

```bash
pip install fiftyone
python download_test_set.py <n_imgs> <out_dir>
```

Você também pode usar outros datasets, se quiser. Só salve as annotations no formato do COCO (https://cocodataset.org/#format-data), e em uma pasta com a estrutura abaixo:
```
<ann_dir>/
	images/
		image1.jpg
		image2.jpg
		...
	annotations.json
```

### Visualizando as anotações

Existem inúmeras APIs para visualizar as anotações, mas eu optei pela implementada no Detectron. Não é exatamente a mais fácil de instalar, mas entre as que eu testei, eu gostei mais dessa, no geral. Você pode usar outras, se preferir, basta modificar a parte de visualização (`personal_lib/plot/`) para utilizar a API desejada.

Requisitos:
* Personal lib:
	```bash
	pip install -e personal-lib
	```
* Plot dependencies
	* [Torch](https://pytorch.org/get-started/locally/)
	* [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

Para rodar:
```bash
python plot_annotations.py -h
```