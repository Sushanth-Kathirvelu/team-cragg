import numpy as np
import urllib
import cv2
import os

class Resize:
    def resize(path,size,imagePath, savePath):
        """function to resize the images
        path -- path of the train.text or file.text
        size -- tuple for the new image dimensions
        imagePath -- path for images or masks
        savePath -- path forsaving the images"""
        if not os.path.exist(savePath):
            os.mkdir(savePath)
        file = open(path)
        for hashNames in file:
            img = cv2.imread(imagePath + "\\" + hashNames +".png")
            img = cv2.resize(img,(size[1],size[0]))
            cv2.imwrite(save_path+ "\\" + hashNames +".png")
