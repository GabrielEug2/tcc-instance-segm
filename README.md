# SegmentationTools

## Instalando tudo

* [Python](https://www.python.org/downloads/)
* [Mask R-CNN](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) (via Detectron)
* [YOLACT](https://github.com/dbolya/yolact)
    * [Esse fork](https://github.com/jerpint/yolact.git) roda na CPU. Pra instalar:
    ```bash
    cd inference
    git clone https://github.com/jerpint/yolact.git
    cd yolact
    git checkout yolact-cpu

    pip install gdown
    gdown https://drive.google.com/u/0/uc?id=1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_&export=download

    pip install Cython
    ```
* [SOLO](https://github.com/aim-uofa/AdelaiDet.git) (via AdelaiDet)
    ```bash
    cd inference
    git clone https://github.com/aim-uofa/AdelaiDet.git
    cd AdelaiDet
    python setup.py build develop

    wget https://cloudstor.aarnet.edu.au/plus/s/chF3VKQT4RDoEqC/download -O SOLOv2_R50_3x.pth
    ```
* [FiftyOne](https://docs.voxel51.com/getting_started/install.html)

## Para usar

* Baixe as imagens com o script:
    ```
    python download_test_set.py <output_dir>
    ```

* Rode os modelos:
    ```bash
    cd inference
    python inference.py ~/input ./results -a
    ```

* Plote as visualizações?