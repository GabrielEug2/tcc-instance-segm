import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PySide6.QtCore import Slot
from segmtools.img_viewer.input_screen import InputScreen
from segmtools.img_viewer.output_screen import OutputScreen

class ImgViewer(QMainWindow):
    @Slot(str)
    def show_detections(self, file_path):
        print(file_path)

    def __init__(self):
        super().__init__()

        self.setWindowTitle('ImgViewer')
        self.resize(230, 150)

        self.inputScreen = InputScreen()
        self.outputScreen = OutputScreen()

        mainWidget = QStackedWidget()
        mainWidget.addWidget(self.inputScreen)
        mainWidget.addWidget(self.outputScreen)
        self.setCentralWidget(mainWidget)

        self.inputScreen.inputArea.image_selected.connect(self.show_detections)


def run():
    app = QApplication(sys.argv)

    mainWindow = ImgViewer()
    mainWindow.show()
    
    app.exec()