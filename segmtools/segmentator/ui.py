import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PySide6.QtCore import Slot

from segmtools.core import ImgInput
from segmtools.core import ImgViewer
from . import backend_logic

class Segmentator(QMainWindow):
    """
    App que recebe uma imagem como entrada, roda nos 3 modelos e mostra os resultados
    """

    @Slot(str)
    def show_predictions(self, img_path):
        predictions = backend_logic.run_on_all_models(img_path)
        original_img = backend_logic.load_img(img_path)

        imgs = [*predictions, original_img]

        self.outputScreen.set_images(imgs)
        self.mainWidget.setCurrentWidget(self.outputScreen)
        self.outputScreen.show_image(0)

    def __init__(self):
        super().__init__()

        self.setWindowTitle('Segmentator')

        # Abre com a janela um pouco pra esquerda, pra facilitar
        # arrastar as imagens do Explorer
        screen_center = self.screen().geometry().center()
        x_offset = 600
        y_offset = 300
        self.move(screen_center.x() - x_offset, screen_center.y() - y_offset)

        self.inputScreen = ImgInput('Arraste uma imagem aqui para rodar nos 3 modelos')
        self.outputScreen = ImgViewer()

        self.mainWidget = QStackedWidget()
        self.mainWidget.addWidget(self.inputScreen)
        self.mainWidget.addWidget(self.outputScreen)
        self.setCentralWidget(self.mainWidget)

        self.mainWidget.setCurrentWidget(self.inputScreen)

        self.inputScreen.imgDropped.connect(self.show_predictions)
        self.outputScreen.imgDropped.connect(self.show_predictions)


def run():
    app = QApplication(sys.argv)

    stylesheet = ""
    with open('./resources/stylesheet.qss') as f:
        stylesheet = f.read()
    app.setStyleSheet(stylesheet)

    segmentator = Segmentator()
    segmentator.show()
    
    sys.exit(app.exec())