import cv2
import numpy as np
from zeroPad import *

class generateNumpy:
    """
    This file generates numpy arrays from the x_train and y_train image data.
    The output of this can be given directly to the model for training.
    """
    def generateNumpy(hash_train_path, hash_test_path, x_train_path,
                      y_train_path, x_test_path, ls=False):
        """
        :hash_path: Path to train.txt or test.txt
        :x_train_path: Path to the x_train images or x_test
        :y_train_path: Path to the y_train images
        :x_test_path: Path to x_test images
        """
        train_zero_pad = []
        masks_zero_pad = []
        test_zero_pad = []
        with open(hash_train_path, 'r') as f:
            myNames = [line.strip() for line in f]
        with open(hash_test_path, 'r') as f:
            myNamesTest = [line.strip() for line in f]
        for file in myNames:
            img = cv2.imread(y_train_path + file + ".png", cv2.IMREAD_GRAYSCALE)
            img = zeroPad(img)
            img = np.array(img)
            masks_zero_pad.append(img)

            img2 = cv2.imread(x_train_path + file + ".png", cv2.IMREAD_GRAYSCALE)
            img2 = zeroPad(img2)
            img2 = np.array(img2)
            train_zero_pad.append(img2)
        for file in myNamesTest:
            img = cv2.imread(x_test_path + file + ".png", cv2.IMREAD_GRAYSCALE)
            img = zeroPad(img)
            img = np.array(img)
            test_zero_pad.append(img)

        masks_zp = np.array(masks_zero_pad)
        train_zp = np.array(train_zero_pad)
        test_zp = np.array(test_zero_pad)
           """
           Reshaping the numpy array to add the channel dimension. Here we're
           working in grayscale which is why we need to explicitly add the
           channel dimension.
           """
        masks_zp = np.reshape(masks_zp, masks_zp.shape + (1,))

        train_zp = np.reshape(train_zp, train_zp.shape + (1,))

        test_zp = np.reshape(test_zp, test_zp.shape + (1,))
            """
            Assuming binary classification. Hence all the cell pixels are
            made 0 as below
            """
        masks_zp[masks_zp == 0]=0
        masks_zp[masks_zp == 1]=0
        masks_zp[masks_zp == 2]=1



        """ Normalization by dividing by 255"""
        train_zp = train_zp / 255
        test_zp = test_zp / 255






        return train_zp, masks_zp, test_zp
