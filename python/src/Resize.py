import numpy as np
import urllib
import cv2
import os

class Resize:
    def resize(path,size,imagePath, savePath):
        """function to resize the images
        path -- ath of the train or file
        size -- tuple for the new image dimensions
        train -- flag for traning and testing dataset (default = True)
        small -- flag for subdata set (default = True)"""
        if not os.path.exist(savePath):
            os.mkdir(savePath)
        file = open(path)
        for hashNames in file:
            img = cv2.imread(imagePath + "\\" + hashNames +".png")
            img = cv2.resize(img,(size[1],size[0]))
            cv2.imwrite(save_path+ "\\" + hashNames +".png")
