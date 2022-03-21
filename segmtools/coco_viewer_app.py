import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PySide6.QtCore import Slot

from segmtools.core import ImgInput
from segmtools.core import ImgViewer
from segmtools.core import utils

class CocoViewer(QMainWindow):
    """
    App que recebe uma imagem como entrada e mostra as respectivas annotations.
    """

    @Slot(str)
    def show_annotations(self, img_path):
        ground_truth = utils.load_ground_truth(img_path)
        raw_img = utils.load_img(img_path)

        imgs = [ground_truth, raw_img]

        self.outputScreen.set_images(imgs)
        self.mainWidget.setCurrentWidget(self.outputScreen)
        self.outputScreen.show_image(0)

    def __init__(self):
        super().__init__()

        self.setWindowTitle('CocoViewer')

        # Abre com a janela um pouco pra esquerda, pra facilitar
        # arrastar as imagens do Explorer
        screen_center = self.screen().geometry().center()
        x_offset = 600
        y_offset = 300
        self.move(screen_center.x() - x_offset, screen_center.y() - y_offset)

        self.inputScreen = ImgInput('Arraste uma imagem aqui para visualizar as anotações do dataset')
        self.outputScreen = ImgViewer()

        self.mainWidget = QStackedWidget()
        self.mainWidget.addWidget(self.inputScreen)
        self.mainWidget.addWidget(self.outputScreen)
        self.setCentralWidget(self.mainWidget)

        self.mainWidget.setCurrentWidget(self.inputScreen)

        self.inputScreen.imgDropped.connect(self.show_annotations)
        self.outputScreen.imgDropped.connect(self.show_annotations)


def run():
    app = QApplication(sys.argv)

    stylesheet = ""
    with open('./resources/stylesheet.qss') as f:
        stylesheet = f.read()
    app.setStyleSheet(stylesheet)

    cocoViewer = CocoViewer()
    cocoViewer.show()
    
    sys.exit(app.exec())