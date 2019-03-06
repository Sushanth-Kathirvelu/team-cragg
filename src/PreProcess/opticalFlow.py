import cv2
import os

class opticalFlow:
	def opticalFlowVideo(textFilePath,createdVideoPath,savePath):
		"""
		this function finds the optical flow from the given video.
        :textFilePath: the path of the text file containing the hashes
					   (train.txt/test.txt)
		:createdVideoPath: the path where the video is saved
        :savePath:  the path where the output is to be saved
		"""
		#Check if the save directory exists, If not create directory
		if not os.path.exists(savePath):
			os.mkdir(savePath)
		#Open the text file
		file = open(textFilePath)

		videoPath = createdVideoPath
		finalMask = savePath

		for hashedData in file:
		    i=0
		    hashedData = hashedData.split("\n")[0]
			#loading two Video
		    cap = cv2.VideoCapture(videoPath + "/" +hashedData + ".avi")
			#reading frame 1
		    ret, frame1 = cap.read()
			#array to PIL
		    frame1 = PIL.Image.fromarray(frame1)
			#Smoothing
		    frame1 = frame1.filter(ImageFilter.SMOOTH)
			#PIL to array
		    frame1 = np.array(frame1)
		    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
		    hsv = np.zeros_like(frame1)
		    hsv[...,1] = 255

		    while(i < 99):
		        ret, frame2 = cap.read()
				#array to PIL
		        frame2 = PIL.Image.fromarray(frame2)
				#Smoothing
		        frame2 = frame2.filter(ImageFilter.SMOOTH)
				#PIL to array
		        frame2 = np.array(frame2)
		        cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
		        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

		        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

		        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
		        hsv[...,0] = ang*180/np.pi/2
		        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
		        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
		        grey = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
		        #Visualize Video
		        cv2.imshow('frame2',rgb)
		        k = cv2.waitKey(30) & 0xff
		        if k == 27:
		            break
		        elif k == 255:
		            print("elif")
		            cv2.imwrite(finalMask + "/" + hashedData +".png",grey)
		        prvs = next
		        i+=1
		    print("out")
		    cap.release()
		    cv2.destroyAllWindows()

	def opticalFlowTwoImage(textFilePath,extractedImgPath,savePath):
		"""
		this function finds the optical flow from the 1st and the 50th frame.
        :textFilePath: the path of the text file containing the hashes
					   (train.txt/test.txt)
		:extractedImgPath: the path where the tar file containing the video
						   frames are extracted
        :savePath:  the path where the output is to be saved
		"""
		#Check if the save directory exists, If not create directory
		if not os.path.exists(savePath):
			os.mkdir(savePath)
		#Open the text file
		file = open(textFilePath)

		imgPath = extractedImgPath
		finalMask = savePath

		for hashedData in file:
		    print(hashedData)
		    i=0
		    hashedData = hashedData.split("\n")[0]
		    #loading two Images
		    frame1 = cv2.imread(imgPath + "/" + hashedData + "/" + "frame0000.png")
		    frame2 = cv2.imread(imgPath + "/" + hashedData + "/" + "frame0050.png")
		    #array to PIL
		    frame1 = PIL.Image.fromarray(frame1)
		    frame2 = PIL.Image.fromarray(frame2)
		    #Smoothing
		    frame1 = frame1.filter(ImageFilter.SMOOTH)
		    frame2 = frame2.filter(ImageFilter.SMOOTH)
		    #PIL t0 image
		    frame1 = np.array(frame1)
		    frame2 = np.array(frame2)
		    #OpticalFlow
		    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
		    hsv = np.zeros_like(frame1)
		    hsv[..., 1] = 255

		    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
		    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
		    hsv[..., 0] = ang * 180 / np.pi / 2
		    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
		    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
		    grey = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)

		    grey[grey>=60] = 2
		    grey[grey<60] = 0

		    cv2.imwrite(finalMask + "/" + hashedData +".png",grey)
