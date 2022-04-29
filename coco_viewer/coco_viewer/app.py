import sys

from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import Slot

from .img_input import ImgInput
from .plot_widget import PlotWidget
from . import plot

class MainWindow(QMainWindow):
    @Slot(str)
    def show_annotations(self, img_path):
        fig = plot.plot_annotations(img_path)

        self.plotWidget = PlotWidget(fig)

        self.plotWidget.show()
        self.plotWidget.windowHandle().requestActivate()

    def __init__(self):
        super().__init__()

        self.setWindowTitle('CocoViewer')

        # Abre com a janela um pouco pra esquerda, pra facilitar
        # arrastar as imagens do Explorer
        self.move(200, 300)

        self.imgInput = ImgInput('Arraste uma imagem aqui para visualizar as anotações do dataset')
        self.imgInput.imgDropped.connect(self.show_annotations)

        self.setCentralWidget(self.imgInput)


def run():
    app = QApplication(sys.argv)

    stylesheet = ""
    with open('./resources/stylesheet.qss') as f:
        stylesheet = f.read()
    app.setStyleSheet(stylesheet)

    mainWindow = MainWindow()
    mainWindow.show()
    
    sys.exit(app.exec())