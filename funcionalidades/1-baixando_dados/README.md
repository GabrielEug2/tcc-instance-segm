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

### Navegando pelas anotações

O COCO usa _um arquivo_ pro _dataset inteiro_. Isso é ótimo em termos de espaço, já que você pode comprimir algumas informações como o nome da imagem e o nome das classes, mas é _horrível_ se você só quer conferir, digamos, quantas anotações de pessoas existem na imagem X.

Aqui, eu prefiro separar as anotações em um arquivo por imagem, contendo só as anotações daquela imagem, e com o nome das classes por extenso, para facilitar a interpretação. Para converter os arquivos para esse formato:

```bash
python split_annotations.py <coco_ann_file> <custom_ann_dir>
```

Daqui em diante, sempre que eu disser "ann_dir", eu estarei me referindo a esse diretório com as anotações no meu formato.

### Visualizando as anotações

Existem inúmeras APIs para visualizar as anotações, mas eu optei pela implementada no Detectron. Não é exatamente a mais fácil de instalar, mas entre as que eu testei, eu gostei mais dessa, no geral. Você pode usar outras, se preferir, basta modificar a parte de visualização (`segm_lib/plot/`) para utilizar a API desejada.

Requisitos:
* [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
	* Instruções no link, mas instalando o meu fork ao invés do padrão, porque eu tive que fazer pequenas modificações:
	```bash
	# python -m pip install 'git+https://github.com/facebookresearch/detectron2.git
	python -m pip install 'git+https://github.com/GabrielEug2/detectron2.git
	```

Para rodar:
```bash
python plot_annotations.py <ann_dir> <img_dir> <out_dir>
```