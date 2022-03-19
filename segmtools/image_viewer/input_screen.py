from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal

class InputArea(QLabel):
    imgDropped = Signal(str)

    def __init__(self):
        super().__init__()

        self.setText('Arraste uma imagem aqui para visualizar as detecções')
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
    

class InputScreen(QWidget):
    imgDropped = Signal(str)

    def __init__(self):
        super().__init__()

        self.inputArea = InputArea()

        layout = QVBoxLayout()
        layout.addWidget(self.inputArea)
        self.setLayout(layout)

        self.inputArea.imgDropped.connect(self.imgDropped)