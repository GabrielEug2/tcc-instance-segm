import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PySide6.QtCore import Slot
from segmtools.image_viewer.input_screen import InputScreen
from segmtools.image_viewer.output_screen import OutputScreen
from segmtools.core import utils

class ImageViewer(QMainWindow):
    @Slot(str)
    def show_predictions(self, img_path):
        raw_img = utils.load_img(img_path)
        ground_truth = utils.load_ground_truth(img_path)
        predictions = utils.load_predictions(img_path)

        imgs = [raw_img, ground_truth, *predictions]
        img_descriptions = ["Imagem original", "Ground truth", "Predições do Mask R-CNN",
                            "Predições do YOLACT", "Predições do SOLO"]

        self.outputScreen.set_imgs(imgs, img_descriptions)
        self.mainWidget.setCurrentWidget(self.outputScreen)
        self.outputScreen.show_first_image()

    def __init__(self):
        super().__init__()

        appSpecificStylesheet = ""
        with open('./resources/image_viewer.qss') as f:
            appSpecificStylesheet = f.read()

        self.setWindowTitle('ImageViewer')
        self.setStyleSheet(appSpecificStylesheet)

        # Abre com a janela um pouco pra esquerda, pra facilitar
        # arrastar as imagens do Explorer
        screen_center = self.screen().geometry().center()
        x_offset = 600
        y_offset = 300
        self.move(screen_center.x() - x_offset, screen_center.y() - y_offset)

        self.inputScreen = InputScreen()
        self.outputScreen = OutputScreen()

        self.mainWidget = QStackedWidget()
        self.mainWidget.addWidget(self.inputScreen)
        self.mainWidget.addWidget(self.outputScreen)
        self.setCentralWidget(self.mainWidget)

        self.mainWidget.setCurrentWidget(self.inputScreen)

        self.inputScreen.imgDropped.connect(self.show_predictions)
        self.outputScreen.imgDropped.connect(self.show_predictions)


def run():
    app = QApplication(sys.argv)

    mainStylesheet = ""
    with open('./resources/segmtools.qss') as f:
        mainStylesheet = f.read()
    
    app.setStyleSheet(mainStylesheet)

    imageViewer = ImageViewer()
    imageViewer.show()
    
    sys.exit(app.exec())