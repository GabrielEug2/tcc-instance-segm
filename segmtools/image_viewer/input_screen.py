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

        event.accept()


class InputScreen(QWidget):
    imgDropped = Signal(str)

    def __init__(self):
        super().__init__()

        self.inputArea = InputArea()

        layout = QVBoxLayout()
        layout.addWidget(self.inputArea)
        self.setLayout(layout)

        # A MainWindow não precisa saber de que widget veio o sinal.
        # Só o que importa é que ele veio da InputScreen.
        self.inputArea.imgDropped.connect(self.imgDropped)