import numpy as np
import urllib
import cv2
import requests
import tarfile
from io import BytesIO

class OFVarMean:
	def OFVarMean(textFilePath,OFImagePath,grayScaleVarianceImagePath,savePath):
		"""
		this function extracts tar file to a specified location .
        :textFilePath: the path of the text file containing the hashes
					   (train.txt/test.txt)
        :OFImagePath: the path where the Optical flow image files are saved
        :grayScaleVarianceImagePath: the path where the Variance image files are saved
        :savePath: the path where the output images is to be saved
		"""
        #Check if the save directory exists, If not create directory
		if not os.path.exists(savePath):
			os.mkdir(savePath)
		#Open the text file
		file = open(textFilePath)

        #Finding the mean of the Optical Flow and Variance Images
        for hashedData in file:
            hashedData = hashedData.split("\n")[0]
            ofImg = cv2.imread(OFImagePath + "\\" + hashedData + ".png")
            varImg = cv2.imread(grayScaleVarianceImagePath + "\\" + hashedData + ".png")
            finalImg = np.add(ofImg,varImg)
            finalImg = finalImg / 2
            cv2.imwrite(savePath + "\\" + hashedData + ".png",finalImg)
