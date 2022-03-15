
from PySide6.QtWidgets import QWidget, QGridLayout, QPushButton

class SelectionScreen(QWidget):
    def __init__(self):
        super().__init__()

        self.resize(800, 400)

        self.button1 = QPushButton("Visualizar imagens e segmentações do dataset")
        self.button1.clicked.connect(open_img_viewer)
        self.button2 = QPushButton("Computar IoU e AP para um conjunto de imagens do dataset")
        self.button2.clicked.connect(open_stats_viewer)
        self.button3 = QPushButton("Computar segmentações para uma imagem qualquer")
        self.button3.clicked.connect(open_segm_viewer)
        
        layout = QGridLayout(self)
        layout.addWidget(self.button1, 0, 0)
        layout.addWidget(self.button2, 0, 1)
        layout.addWidget(self.button3, 1, 0, 1, 2)
        self.setLayout(layout)

    def open_img_viewer():
        pass

    def open_stats_viewer():
        pass

    def open_segm_viewer():
        pass