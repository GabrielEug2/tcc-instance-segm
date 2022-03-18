import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PySide6.QtCore import Slot
from segmtools.img_viewer.input_screen import InputScreen
from segmtools.img_viewer.output_screen import OutputScreen
from segmtools.core import utils

class ImgViewer(QMainWindow):
    @Slot(str)
    def show_predictions(self, img_path):
        raw_img = utils.load_img(img_path)
        ground_truth = utils.load_ground_truth(img_path)
        predictions = utils.load_predictions(img_path)

        new_height, new_width, _ = raw_img.shape
        new_height += 30
        print(new_width, new_height)
        print(self.size())
        self.resize(new_width, new_height)
        print(self.size())

        self.outputScreen.set_images([raw_img, ground_truth, *predictions])
        self.mainWidget.setCurrentWidget(self.outputScreen)

    def __init__(self):
        super().__init__()

        self.setWindowTitle('ImgViewer')
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setStyleSheet('''
            background-color: #242a30;
            color: #b6c1cc;
            font-size: 15px;
        }''')
        self.inputScreen = InputScreen()
        self.outputScreen = OutputScreen()

        self.mainWidget = QStackedWidget()
        self.mainWidget.addWidget(self.inputScreen)
        self.mainWidget.addWidget(self.outputScreen)
        self.setCentralWidget(self.mainWidget)

        self.mainWidget.setCurrentWidget(self.inputScreen)
        self.resize(530, 340)

        screen_center = self.screen().geometry().center()
        x_offset = 600
        y_offset = 300
        self.move(screen_center.x() - x_offset, screen_center.y() - y_offset)

        self.inputScreen.imgDropped.connect(self.show_predictions)
        self.outputScreen.imgDropped.connect(self.show_predictions)


def run():
    app = QApplication(sys.argv)

    mainWindow = ImgViewer()
    mainWindow.show()
    
    sys.exit(app.exec())