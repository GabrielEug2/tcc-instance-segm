# SegmentationTools

## COCO Viewer

Para visualizar as anotações do COCO.

Como usar:

* Baixe as imagens e as annotations no site oficial do COCO (https://cocodataset.org/#download).
* Instale as dependências.
    ```bash
    pip install pycocotools
    pip install PySide6 opencv-python matplotlib pyyaml
    ```
* Altere o arquivo `coco_viewer/config.yaml` para informar ao programa a localização das annotations.
* Rode o programa e siga as instruções.
    ```bash
    cd coco_viewer
    python coco_viewer.py
    ```


## Inference

Para rodar uma imagem em vários modelos e comparar os resultados.

Como usar:

* Instale as dependências.
    ```bash
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    pip install opencv-python
    ```

* Instale os modelos.
    * Mask R-CNN
        ```bash
        sudo apt install gcc g++
        sudo apt install ninja-build
        python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
        ```

    * YOLACT
        ```bash
        cd inference
        git clone https://github.com/jerpint/yolact.git
        cd yolact
        git checkout yolact-cpu

        pip install gdown
        gdown https://drive.google.com/u/0/uc?id=1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_&export=download

        conda install cython
        ```

    * SOLO
        ```bash
        cd inference
        git clone https://github.com/aim-uofa/AdelaiDet.git
        cd AdelaiDet
        python setup.py build develop

        wget https://cloudstor.aarnet.edu.au/plus/s/chF3VKQT4RDoEqC/download -O SOLOv2_R50_3x.pth
        ```

* Rode o programa nas imagens desejadas.
    ```bash
    cd inference
    python inference.py ~/input ./results
    ```

* Para ver a lista completa de parâmetros:
    ```
    python inference.py -h
    ```