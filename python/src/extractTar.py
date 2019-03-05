import cv2
import os
import tarfile

class extractTar:
	def extractTar(textFilePath,extractedPath):
		"""
		this function extracts tar file to a specified location .
        :textFilePath: the path of the text file containing the hashes
					   (train.txt/test.txt)
        :extractedPath:  the path where the extracted tar file is to be saved
		"""
		#Check if the save directory exists, If not create directory
		if not os.path.exists(createdVideoPath):
			os.mkdir(createdVideoPath)
		#Open the text file
		file = open(textFilePath)

		for hashedData in file:
			hashedData = hashedData.split("\n")[0]
			tar = tarfile.open(extractedPath+ "\\" + hashedData + ".tar")
			#extract the contents of the tar file to the given location
			tar.extractall(path = extractedPath)
			tar.close()
