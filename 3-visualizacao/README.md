
## Visualização

Requisitos:
* Linux - talvez você consiga instalar no Windows, mas eu não testei.
* Algumas imagens e anotações do OpenImages, e a saída dos modelos para essas
imagens. Você pode ou baixar as imagens você mesmo e rodar a inferência, ou só
usar os exemplos em `sample_data`.
* [Detectron](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
    * Instruções no link (não sei se roda no Windows, eu instalei no Linux)
* [COCO Api](https://github.com/cocodataset/cocoapi.git)
    ```bash
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonApi
    make
    ```
* Minha biblioteca de visualização que usa tudo isso:
    ```bash
    # Não precisa instalar a biblioteca em si, só as dependências
    pip install -r requirements.txt
    ```

Como usar:
* `python fix_annotations.py <ann_file>` (não precisa se estiver usando os exemplos)
* `python plot.py -h`