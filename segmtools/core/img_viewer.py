
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal

class ImgViewer(QWidget):
    """Widget para visualização de imagens. Para configurar as imagens que
    serão exibidas, use o método `set_images`."""

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

        self.images = []

        self.setFocusPolicy(Qt.StrongFocus)
        self.setAcceptDrops(True)

    def set_images(self, images):
        self.images = images

    def show_image(self, i):
        valid_index = i in range(0, len(self.images))

        if valid_index:
            # Mostra a respectiva imagem
            requested_image = self.images[i]

            self.imgArea.setPixmap(requested_image.to_pixmap())
            self.imgDescription.setText(requested_image.description)
            self.currentIndex = i
        else:
            # Fica na mesma
            pass

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

    def show_next_image(self):
        self.show_image(self.currentIndex + 1)
        
    def show_previous_image(self):
        self.show_image(self.currentIndex - 1)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        img_path = event.mimeData().urls()[0].toLocalFile()

        self.imgDropped.emit(img_path)

        event.accept()
