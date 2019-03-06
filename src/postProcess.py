
import numpy as np
import urllib
import cv2
import requests
import tarfile
from io import BytesIO
import skimage.io as io
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
class postProcess:
    def predict(path,var,model,savePath,zeroPadd = True, size = None):

        """function to predict the y_yest
        path -- path of the train.text or file.text
        var -- numpy array for y_test
        model -- loaded model
        savePath -- path forsaving the images
        zeroPadd = flag weather the data set is zero padded or not
        (default=True)"""

        if not os.path.exist(savePath):
            os.mkdir(savePath)
        file = open(path)
        i=0
        for im in var:
            hashName = file[i].strip()
            og_columns = im.shape[1]
            og_rows = im.shape[0]
            if zeroPadd:
                image = zero_padding(im)
            else:
                image = cv2.resize(image,(size[1],size[0]))
            image = image[np.newaxis,...]
            image = image[...,np.newaxis]
            mask = model.predict(image)
            mask = mask[0,...]
            mask = mask[...,0]
            if zeroPadd:
                mask = mask[0:og_rows, 0:og_columns]
            else:
                mask = cv2.resize(mask,(og_columns,og_rows))
            for x in range(0,mask.shape[0]):
                for y in range(0,mask.shape[1]):
                    if mask[x,y] >= 0.6:
                        mask[x,y] = 2
                    else:
                        mask[x,y] = 0
            cv2.imwrite(savePath + "/" + hashName + ".png", mask)
            print(".......................................",i)
            i += 1
