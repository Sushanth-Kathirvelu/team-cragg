import cv2
import os

class generateVideo:
	def generateVideo(textFilePath,extractedPath,createdVideoPath):
		"""
		this function generates a video from the extracted tar file .
        :textFilePath: the path of the text file containing the hashes
					   (train.txt/test.txt)
        :extractedPath: the path where the corresponding tar files are extracted
        :createdVideoPath: the path where the video is to be saved
		"""
		#Check if the save directory exists, If not create directory
		if not os.path.exists(createdVideoPath):
			os.mkdir(createdVideoPath)
		#Open the text file
		file = open(textFilePath)
		videoPath = createdVideoPath

		for hashedData in file:
			hashedData = hashedData.split("\n")[0]
			image_folder = extractedPath + "/" + "data" + "/" + hashedData
			video_name = hashedData + ".avi"
			images = os.listdir(image_folder)
			frame = cv2.imread(os.path.join(image_folder, images[0]))
			height, width, layers = frame.shape
			#declare the video writter
			video = cv2.VideoWriter(videoPath + "/" +video_name, 0, 1, (width,height))
			#Write all images to a single video
			for image in images:
				video.write(cv2.imread(os.path.join(image_folder, image)))

			cv2.destroyAllWindows()
			video.release()
