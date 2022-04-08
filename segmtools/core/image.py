
from PySide6.QtGui import QImage, QPixmap

class Image():
    """Classe que representa uma imagem plotável no ImgViewer. Além da imagem
    em si, tem que ter uma descrição pra mostrar junto ("Imagem original",
    "Predictions do modelo tal" ou algo assim)"""

    def __init__(self, img, description):
        self.img = img
        self.description = description

    def to_pixmap(self):
        height, width, channels = self.img.shape
        bytes_per_line = channels * width

        pixmap = QPixmap(QImage(self.img, width, height, bytes_per_line, QImage.Format_RGB888))

        return pixmap