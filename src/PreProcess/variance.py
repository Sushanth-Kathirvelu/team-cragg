import numpy as np
import cv2
import os

class variance:

    def calculateGrayscaleVariance(path,data_set,savePath):

        ''' function to calculate variaince over a dataset.
        Keyword arguments:
        path -- train.txt or test.txt
        data set -- the data set in th form of a list of np arrays over which
        variance has to be calculated
        savePath -- path for saving the grayscale variance data_set

        '''

        if not os.path.exist(savePath):
            os.mkdir(savePath)

        ## path for the traing file

        file = open(path)
        for i in range(len(data_set)):
            hashName = file[i].strip()
            variances = np.var(data_set[i],axis=0)
            im = (variances/np.max(variances))
            for x in range(0,im.shape[0]):
                for y in range(0,im.shape[1]):
                    if im[x,y] >= 0.5:
                        im[x,y] = 1
                    else:
                        im[x,y] = im[x,y]*1

            cv2.imwrite(savePath +"/"+ hashName +".png", im)
