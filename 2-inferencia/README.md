
## Inferência

Requisitos:
* Linux - talvez você consiga instalar no Windows, mas eu não testei.
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
* [COCO Api](https://github.com/cocodataset/cocoapi.git)
    ```bash
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonApi
    make
    ```
* Outras dependencias do meu código:
    ```bash
	pip install opencv-python pyyaml
    ```

Como usar:
* Edite no arquivo `config.yaml` o local onde você instalou os modelos.
* `python inference.py <img_dir> <out_dir>`