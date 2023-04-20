## Inferencia e visualização

### Inferência

Requisitos:
* Linux - talvez você consiga instalar no Windows, mas eu não testei.
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
	* Para usar com outros modelos, veja `personal_lib/inference/predictors/base_predictor.py` para a interface necessária.
* Partes relevantes da minha biblioteca:
	```bash
	pip install -e personal-lib/personal-lib-pred-logic
	pip install -e personal-lib/personal-lib-core
	pip install -e personal-lib/personal-lib-plot # Precisa do Detectron. Leia abaixo.
	pip install -e personal-lib/personal-lib-inference
	```
* Plot lib
	* Existem inúmeras APIs para visualizar as predictions, mas eu optei pela implementada no [Detectron](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). Não é exatamente a mais fácil de instalar, mas entre as que eu testei, eu gostei mais dessa, no geral. Você pode usar outras, se preferir, basta modificar a parte de visualização (`personal-lib/personal-lib-plot/`) para utilizar a API desejada. Assim você não precisa instalar o Detectron, se não quiser.
	* Third party
		* [Torch](https://pytorch.org/get-started/locally/)
		* [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
			* Instruções no link
	* Partes relevantes da biblioteca:
		```bash
		pip install -e personal-lib/personal-lib-pred-logic
		pip install -e personal-lib/personal-lib-core
		pip install -e personal-lib/personal-lib-plot
		```

Como usar:
* Edite no arquivo `personal-lib/personal-lib-inference/personal_lib/inference/predictors/config.yaml` o local onde você instalou os modelos. É, eu sei. Eu vou simplificar esse processo depois.
* `python inference.py <img_file_or_dir> <out_dir>` para rodar a inferência.
* `python inference.py -h` para mais opções.