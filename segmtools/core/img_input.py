
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt, Signal

class ImgInput(QLabel):
    """Widget no qual é possível soltar imagens. Quando isso acontece,
    emite um sinal com o caminho da imagem."""
    
    imgDropped = Signal(str)

    def __init__(self, text):
        super().__init__()

        self.setText(text)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignCenter)

        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        img_path = event.mimeData().urls()[0].toLocalFile()

        self.imgDropped.emit(img_path)

        event.accept()