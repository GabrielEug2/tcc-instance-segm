
## Analises e visualização

Requisitos:
* Algumas imagens e anotações do OpenImages, e a saída dos modelos para essas
imagens. Você pode ou baixar as imagens você mesmo e rodar a inferência, ou só
usar os exemplos em `sample_data`.
* [FiftyOne](https://docs.voxel51.com/getting_started/install.html)
* [COCO Api](https://github.com/cocodataset/cocoapi.git)

    Windows:
    ```posh
    pip install pycocotools
    ```

    Linux:
    ```bash
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonApi
    make
    ```

Como usar:
* Coloque as imagens e as predictions em uma pasta com a seguinte estrutura:
```
<dataset_dir>/
    images/
        img-1.jpg
        img-2.jpg
        ...
    annotations.json
<predictions_dir>/
    maskrcnn_pred.json
    solo_pred.json
    yolact_pred.json
```
* `python eval.py --dataset_dir=<dataset_dir> --predictions_dir=<predictions_dir>` (por padrão, usa os dados em sample_data)