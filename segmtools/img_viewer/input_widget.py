

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import Qt

from output_widget import OutputWidget


class InputMessage(QLabel):
    def __init__(self):
        super().__init__()

        self.setAcceptDrops(True)

        self.setAlignment(Qt.AlignCenter)
        self.setText('Arraste uma imagem aqui para visualizar as detecções')
        self.setStyleSheet('''
            QLabel{
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
            event.accept()
            event.setDropAction(Qt.CopyAction)

            file_path = event.mimeData().urls()[0].toLocalFile()
            print(f"\n\n{file_path}\n\n")

            # raise signal to main window
            # how?
        else:
            event.ignore()

class InputWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()

        self.inputBox = InputMessage()
        layout.addWidget(self.inputBox)

        self.setLayout(layout)

        self.show()