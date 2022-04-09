
from PySide6.QtWidgets import QStackedWidget

# O ImgInput e o ImgViewer são dois widgets de tamanhos diferentes. Por algum
# motivo, o QStackedWidget prefere ficar sempre com o tamanho do maior widget,
# mesmo quando está exibindo o outro.
# 
# Essa é uma maneira de resolver isso.
#   https://stackoverflow.com/questions/23511430/qt-qstackedwidget-resizing-issue
class StackedWidget(QStackedWidget):
    def sizeHint(self):
        return self.currentWidget().sizeHint()

    def minimumSizeHint(self):
        return self.currentWidget().minimumSizeHint()