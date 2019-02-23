from keras.preprocessing.image import *

#maintain a list of hashes (filenames)
with open("/home/rutu/DSP/test.txt","r") as f:
    myNames = [line.strip() for line in f]
        
#maintain a list of all shapes of the test images
list1 = []

for file in myNames:
    image = cv2.imread("/home/rutu/DSP/test_frame0000/"+file+".png", cv2.IMREAD_GRAYSCALE)
    list1.append(image.shape)

#load and convert the saved numpy array of masks to .png
print("array to image")
imgs = np.load('/home/rutu/DSP/imgs_mask_test.npy')

i=0
for fileName in myNames:
    img = imgs[i]
    print(np.unique(img))
    #resize back to original size of image
    img = cv2.resize(img, (list1[i][1], list1[i][0]))
    img[img<=0.5] = 0
    img[img>0.5] = 2
    cv2.imwrite("/home/rutu/DSP/results_resized1/"+ fileName +".png",img)
    i=i+1
