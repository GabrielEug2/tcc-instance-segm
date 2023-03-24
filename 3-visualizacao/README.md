
## Visualização

* plot_predictions --masks-too
* plot_annotations

Requisitos:
* Algumas imagens e anotações do OpenImages, e a saída dos modelos para essas
imagens. Você pode ou baixar as imagens você mesmo e rodar a inferência, ou só
usar os exemplos em `sample_data`.
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
* `python evaluate.py <img_dir> <ann_file> <pred_dir>` (por padrão, usa os dados em sample_data)