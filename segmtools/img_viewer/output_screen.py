from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal

from segmtools.core import utils

class ImageArea(QWidget):
    def __init__(self):
        super().__init__()


        layout = QVBoxLayout()
        layout.addWidget(self.currentImage)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
    
    def show_image(self, index):
        valid_index = index in range(0, len(self.images))

        if valid_index:
            # plota
            img = self.images[index]
            pixmap = utils.numpy_to_pixmap(img)
            self.currentImage.setPixmap(pixmap)
            self.current_index = index
        else:
            # fica na mesma
            pass

    def next(self):
        # raw_img.setPixmap(QPixmap(img_path))
        # self.currentIndex += 1
        pass
    
    def previous(self):
        # raw_img.setPixmap(QPixmap(img_path))
        # self.currentIndex -= 1
        pass


class OutputScreen(QWidget):
    imgDropped = Signal(str)

    def __init__(self):
        super().__init__()

        self.imageArea = QLabel()
        self.controls = QLabel('Use as setas para navegar entre as imagens')
        self.controls.setFixedHeight(30)
        self.controls.setAlignment(Qt.AlignCenter)
        
        self.images = []
        self.currentIndex = 0

        layout = QVBoxLayout()
        layout.addWidget(self.imageArea)
        layout.addWidget(self.controls)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.setAcceptDrops(True)
    
    def set_images(self, images):
        self.images = images
        self.show_image(0)

    def show_image(self, index):
        valid_index = index in range(0, len(self.images))

        # Só plota se for um índice válido, senão fica na mesma
        if valid_index:
            img = self.images[index]
            pixmap = utils.numpy_to_pixmap(img)

            self.imageArea.setPixmap(pixmap)
            self.current_index = index

    def next(self):
        # raw_img.setPixmap(QPixmap(img_path))
        # self.currentIndex += 1
        pass
    
    def previous(self):
        # raw_img.setPixmap(QPixmap(img_path))
        # self.currentIndex -= 1
        pass

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()
    
    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            img_path = event.mimeData().urls()[0].toLocalFile()

            self.imgDropped.emit(img_path)

            event.accept()
        else:
            event.ignore()