## Inferencia

Requisitos:
* Linux - talvez você consiga instalar os modelos no Windows, mas eu não testei.
* Modelos:
	* [Mask R-CNN](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
		* Instruções no link
	* [YOLACT](https://github.com/dbolya/yolact)
		* [Esse fork](https://github.com/jerpint/yolact.git) roda na CPU. Pra instalar:
		```bash
		git clone https://github.com/jerpint/yolact.git yolact_pkg
		cd yolact_pkg
		git checkout yolact-cpu

		pip install gdown
		gdown https://drive.google.com/u/0/uc?id=1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_&export=download

		pip install Cython
		cd -
		```
	* [SOLO](https://github.com/aim-uofa/AdelaiDet.git)
		```bash
		git clone https://github.com/aim-uofa/AdelaiDet.git
		cd AdelaiDet
		python setup.py build develop

		wget https://cloudstor.aarnet.edu.au/plus/s/chF3VKQT4RDoEqC/download -O SOLOv2_R50_3x.pth
		cd -
		```
	* Para usar com outros modelos, veja `personal-lib/_lib/inference/predictors/base_predictor.py` para a interface necessária.
* Personal lib:
	```bash
	pip install -e personal-lib
	```

Como usar:
* Edite no arquivo `personal-lib/personal_lib/inference/predictors/config.yaml` o local onde você instalou os modelos. É, eu sei. Eu vou simplificar esse processo depois.
* `python inference.py <img_file_or_dir> <out_dir>`

### Visualizando as predictions

Existem inúmeras APIs para visualizar as predictions em segmentação de instâncias, mas eu optei pela implementada no Detectron. Não é exatamente a mais fácil de instalar, mas entre as que eu testei, eu gostei mais dessa, no geral. Você pode usar outras, se preferir, basta modificar a parte de visualização (`personal_lib/plot/`) para utilizar a API desejada.

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
python plot_predictions.py -h
```