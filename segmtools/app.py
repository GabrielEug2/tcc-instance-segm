
import sys
from PySide6.QtWidgets import QApplication, QWidget

class SelectionScreen(QWidget):
    def __init__(self):
        super().__init__()

        self.resize(800, 400)


from PySide6.QtWidgets import QWidget, QGridLayout, QPushButton

class SelectionWindow(QWidget):
    def __init__(self):
        self.button1 = QPushButton("Visualizar imagens e segmentações do dataset")
        self.button2 = QPushButton("Computar IoU e AP para um conjunto de imagens do dataset")
        self.button3 = QPushButton("Computar segmentações para uma imagem qualquer")
        
        layout = QGridLayout(self)
        layout.addWidget(self.button1, 0, 0)
        layout.addWidget(self.button2, 0, 1)
        layout.addWidget(self.button3, 1, 0, 1, 2)
        self.setLayout(layout)

        
def run():
    app = QApplication(sys.argv)

    mainWindow = SelectionScreen()
    mainWindow.show()

    exit_code = app.exec()
    return exit_code