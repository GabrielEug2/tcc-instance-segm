
## Inferência

Requisitos:
* Linux
* [Mask R-CNN](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
    * Instruções no link
* [YOLACT](https://github.com/dbolya/yolact)
    * [Esse fork](https://github.com/jerpint/yolact.git) roda na CPU. Pra instalar:
    ```bash
    cd inferencia
    git clone https://github.com/jerpint/yolact.git yolact_pkg
    cd yolact_pkg
    git checkout yolact-cpu

    pip install gdown
    gdown https://drive.google.com/u/0/uc?id=1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_&export=download

    pip install Cython
    ```
* [SOLO](https://github.com/aim-uofa/AdelaiDet.git)
    ```bash
    cd inferencia
    git clone https://github.com/aim-uofa/AdelaiDet.git
    cd AdelaiDet
    python setup.py build develop

    wget https://cloudstor.aarnet.edu.au/plus/s/chF3VKQT4RDoEqC/download -O SOLOv2_R50_3x.pth
    ```
* [COCO Api](https://github.com/cocodataset/cocoapi.git)
    ```bash
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonApi
    make
    ```
* Minha biblioteca de inferência que usa tudo isso:
    ```bash
    pip install -r requirements.txt
    pip install -e inference_lib
    ```

Como usar:
* Edite no arquivo `config.yaml` o local onde você instalou os modelos.
* `python inference.py <img_dir> <out_dir>`