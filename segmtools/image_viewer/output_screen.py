
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, Signal

from segmtools.core import utils

class OutputScreen(QWidget):
    imgDropped = Signal(str)

    def __init__(self):
        super().__init__()

        self.imgArea = QLabel()
        self.imgDescription = QLabel()
        self.imgDescription.setObjectName('imgDescription')

        self.imgArea.setAlignment(Qt.AlignCenter)
        self.imgDescription.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.imgArea)
        layout.addWidget(self.imgDescription)
        self.setLayout(layout)

        self.imgs = []
        self.imgDescriptions = []

        self.setFocusPolicy(Qt.StrongFocus)
        self.setAcceptDrops(True)

    def set_imgs(self, imgs, img_descriptions):
        self.imgs = imgs
        self.imgDescriptions = img_descriptions

    def show_image(self, i):
        valid_index = i in range(0, len(self.imgs))

        if valid_index:
            # Mostra a respectiva imagem
            img = self.imgs[i]
            description = self.imgDescriptions[i]

            pixmap = utils.numpy_to_pixmap(img)

            self.imgArea.setPixmap(pixmap)
            self.imgDescription.setText(description)
            self.currentIndex = i
        else:
            # Fica na mesma
            pass

    def show_first_image(self):
        self.show_image(0)

    def show_next_image(self):
        self.show_image(self.currentIndex + 1)
        
    def show_previous_image(self):
        self.show_image(self.currentIndex - 1)

    def keyPressEvent(self, event):
        match event.key():
            case Qt.Key_Right | Qt.Key_Down:
                event.accept()
                self.show_next_image()
            case Qt.Key_Left | Qt.Key_Up:
                event.accept()
                self.show_previous_image()
            case _:
                event.ignore()

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        img_path = event.mimeData().urls()[0].toLocalFile()

        self.imgDropped.emit(img_path)

        event.accept()
