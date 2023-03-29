
## Visualização

Requisitos:
* Algumas imagens e anotações do OpenImages, e a saída dos modelos para essas
imagens. Você pode ou baixar as imagens você mesmo e rodar a inferência, ou só
usar os exemplos em `sample_data`.
* [Detectron](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
    * Instruções no link (não sei se roda no Windows, eu instalei no Linux)
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
* `python fix_annotations.py <ann_file>` (não precisa se estiver usando os exemplos)
* `python plot_annotations.py <img_file> <ann_file> <out_dir>`
* `python plot_predictions.py <img_file> <pred_file> <out_dir>`