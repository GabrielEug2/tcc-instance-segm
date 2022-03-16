from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, Signal, Slot

class OutputScreen(QWidget):
    imgDropped = Signal(str)

    @Slot(str)
    def show_detections(self, img_path):
        # TODO: matplotlib plots --> image
        # 1st img = open(img_path)
        # 2nd img = ground_truth
        # 3rd/4th/5th = predictions
        print(img_path)

    def __init__(self):
        super().__init__()

        # TODO: Gallery app
        # Começa mostrando a imagem normal, e muda pras outras com as setinhas

        mainLayout = QVBoxLayout()
        self.setLayout(mainLayout)

        self.setAcceptDrops(True)

    def next():
        pass
    
    def previous():
        pass

# =====================================================================
# TO DO: plotar o resultado com a própria API do Detectron e 
#        as ground truths com a API do COCO
# =====================================================================

# =====================================================================

# %matplotlib inline
# from pycocotools.coco import COCO
# import numpy as np
# import skimage.io as io
# import matplotlib.pyplot as plt
# import pylab
# pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# dataDir='..'
# dataType='val2017'
# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# # initialize COCO api for instance annotations
# coco=COCO(annFile)

# # load and display image
# # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# # use url to load image
# I = io.imread(img['coco_url'])
# plt.axis('off')
# plt.imshow(I)
# plt.show()

# # load and display instance annotations
# plt.imshow(I); plt.axis('off')
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco.loadAnns(annIds)
# coco.showAnns(anns)



# # =====================================================================


# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog

# test_img = os.listdir(OUTPUT_DIR):
# img_filename = os.path.basename(results_file).rstrip('.json') + '.jpg'

# img = cv2.imread(os.path.join(IMG_DIR, filename) # BGR


# base.results_file

# with open(filename) as f:
#     detections = json.load(f)

# img_filename
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# v = Visualizer(img, MetadataCatalog.get('teste'), scale=1.2)
# v = v.draw_instance_predictions(outputs['instances'].to('cpu'))