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

* Instale os modelos que você deseja fazer inferência.
    * Mask R-CNN

        Precisa instalar mesmo que não vá usar. Eu uso a API deles pra plotar os resultados.

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
        ```

* Rode o programa nas imagens desejadas.
    ```bash
    cd inference
    python inference.py ~/input ./results
    ```

* Use (`python inference.py -h`) para ver a lista completa de parâmetros.


## Temp

Eu preciso fazer os 3 modelos rodarem na CPU. 
A versão do PyTorch não importa, é só usar environments diferentes se precisar de uma versão específica.
Como alguns modelos não funcionam no Windows, eu preciso rodar tudo no Subsistema do Windows pra Linux (WSL).