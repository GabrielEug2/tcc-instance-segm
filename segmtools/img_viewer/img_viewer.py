import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PySide6.QtCore import Slot
from segmtools.img_viewer.input_screen import InputScreen
from segmtools.img_viewer.output_screen import OutputScreen

class ImgViewer(QMainWindow):
    @Slot(str)
    def show_detections(self, img_path):
        self.mainWidget.setCurrentWidget(self.outputScreen)
        self.resize(400, 400)
        
        self.outputScreen.show_detections(img_path)

    def __init__(self):
        super().__init__()

        self.setWindowTitle('ImgViewer')
        self.resize(530, 340)

        self.inputScreen = InputScreen()
        self.outputScreen = OutputScreen()

        self.mainWidget = QStackedWidget()
        self.mainWidget.addWidget(self.inputScreen)
        self.mainWidget.addWidget(self.outputScreen)
        self.setCentralWidget(self.mainWidget)

        self.inputScreen.imgDropped.connect(self.show_detections)
        self.outputScreen.imgDropped.connect(self.show_detections)


def run():
    app = QApplication(sys.argv)

    mainWindow = ImgViewer()
    mainWindow.show()
    
    sys.exit(app.exec())