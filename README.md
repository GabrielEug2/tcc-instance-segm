# SegmentationTools

### coco_viewer

Para visualizar as anotações do COCO.

Como usar:

* Baixe as imagens e as annotations no site oficial do COCO (https://cocodataset.org/#download).

* Instale as dependências.
    ```bash
    pip install PySide6
    pip install opencv-python
    conda install pyyaml
    ```

* Altere o arquivo `coco_viewer/config.yaml` para informar ao programa a localização das annotations.

* Rode o programa e siga as instruções.
    ```bash
    cd coco_viewer
    python coco_viewer.py
    ```

### inference

Para rodar uma imagem em vários modelos e comparar os resultados.

Como usar:

* Instale os modelos que você deseja fazer inferência.

    * Mask R-CNN
        ```bash
        # PyTorch
        conda install pytorch torchvision torchaudio cpuonly -c pytorch

        # O modelo em si
        sudo apt install gcc g++
        sudo apt install ninja-build
        python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
        ```

    * YOLACT


    * SOLO



* Instale as dependências.
    ```bash
    conda install pyyaml
    pip install opencv-python
    ```

* Execute os modelos.
    ```bash
    python mask_rcnn.py '<img_path>'
    python yolact.py '<img_path>'
    python solo.py '<img_path>'
    ```

## Temp

Eu preciso fazer os 3 modelos rodarem na CPU. 
A versão do PyTorch não importa, é só usar environments diferentes se precisar de uma versão específica.
Como alguns modelos não funcionam no Windows, eu preciso rodar tudo no Subsistema do Windows pra Linux (WSL).