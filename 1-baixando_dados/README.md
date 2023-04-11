## Baixando dados

### Do Openimages

Para esse projeto, eu usei dados do Openimages, com algumas das classes que mais aparecem no COCO. Você pode baixar uma dataset com essas mesmas características seguindo os passos abaixos.

```bash
pip install fiftyone
python download_test_set.py <n_imgs> <out_dir>
```

### De outras fontes

Eu usei dados como eu descrevi acima, mas você pode usar qualquer dataset. Só salve as annotations no formato do COCO (https://cocodataset.org/#format-data), e em uma pasta com a estrutura abaixo:
```
<ann_dir>/
	images/
		image1.jpg
		image2.jpg
		...
	annotations.json
```

### Visualizando as anotações

Existem inúmeras APIs para visualizar as anotações, mas eu optei pela implemetada no Detectron. Não é exatamente a mais fácil de instalar, mas entre as que eu testei, eu gostei mais dessa, no geral. Você pode usar outra, se preferir.

Requisitos:
* Linux - talvez você consiga instalar no Windows, mas eu não testei.
* [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
    * Instruções no link
* [COCO Api](https://github.com/cocodataset/cocoapi.git)
    ```bash
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonApi
    make
    ```
* Plot lib
	```bash
	pip install -e ../personal_lib/plot
	```

Para rodar:
```bash
python plot_annotations.py <ann_dir> <out_dir>
```