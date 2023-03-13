# SegmentationTools

Tudo precisa de Python pra rodar, então instale isso primeiro.

## Misc 

Scripts aleatórios.

* `class_dist_coco.py`: calcula a distribuição de classes nos objetos do COCO.
    * Requisitos:
        * Anotações do COCO ([train/val 2017](https://cocodataset.org/#download))

    * Como usar:
        * Edite no arquivo o caminho das annotations.
        * `python class_dist_coco.py`

* `download_test_set.py`: baixa 200 imagens e anotações do OpenImages, com algumas das classes que tem mais objetos no COCO.
    * Requisitos:
        * [FiftyOne](https://docs.voxel51.com/getting_started/install.html)

    * Como usar:
        * `python download_test_set.py <output_dir>`

## Inferência

Requisitos:
* [Mask R-CNN](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
    * Instruções no link
* [YOLACT](https://github.com/dbolya/yolact)
    * [Esse fork](https://github.com/jerpint/yolact.git) roda na CPU. Pra instalar:
    ```bash
    git clone https://github.com/jerpint/yolact.git
    cd yolact
    git checkout yolact-cpu

    pip install gdown
    gdown https://drive.google.com/u/0/uc?id=1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_&export=download

    pip install Cython
    ```
* [SOLO](https://github.com/aim-uofa/AdelaiDet.git)
    ```bash
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
* Outras dependências (Opencv, tqdm...):
```bash
python -m pip install -r segmentation-tools/inferencia/requirements.txt
```

Como usar:
```bash
cd inferencia
python inference.py <input_dir> <output_dir>
```

## Analises e visualização

Requisitos:
* Algumas imagens e anotações do OpenImages (ver acima)
* A saída dos modelos para essas imagens (tem alguma de teste na pasta)
* [FiftyOne](https://docs.voxel51.com/getting_started/install.html)
