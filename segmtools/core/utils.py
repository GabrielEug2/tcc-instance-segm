
import cv2
import numpy as np

def open_as_rgb(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def create_black_img(shape):
    black_img = np.zeros(shape, dtype=np.uint8)
    return black_img