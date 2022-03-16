
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import Qt, Signal


class InputArea(QLabel):
    image_selected = Signal(str)

    def __init__(self, parent):
        super().__init__(parent)

        self.setAcceptDrops(True)

        self.setText('Arraste uma imagem aqui para visualizar as detecções')
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setMargin(10)
        self.setStyleSheet('''
            InputArea {
                border: 4px dashed #aaa
            }
        ''')

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
            file_path = event.mimeData().urls()[0].toLocalFile()

            self.image_selected.emit(file_path)

            event.accept()
        else:
            event.ignore()

class InputScreen(QWidget):
    def __init__(self):
        super().__init__()
      
        self.inputArea = InputArea(self)

        layout = QVBoxLayout()
        layout.addWidget(self.inputArea)
        self.setLayout(layout)