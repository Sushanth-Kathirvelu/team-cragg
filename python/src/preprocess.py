
import numpy as np
import urllib
import cv2
import requests
import tarfile
from io import BytesIO
from collections import Counter

# train file path
url = "https://storage.googleapis.com/uga-dsp/project2/train.txt"
train_text_file_list = requests.get(url).text
train_text_file_list = train_text_file_list.split()
# test file path
url2 = "https://storage.googleapis.com/uga-dsp/project2/test.txt"
test_text_file_list = requests.get(url2).text
test_text_file_list = test_text_file_list.split()


# extracting the all training images from each file as numpy arrays
all_file_images = list()
for filename in train_text_file_list:
    url3 = "https://storage.googleapis.com/uga-dsp/project2/data/"+filename+".tar"
    response = requests.get(url3)
    tar = tarfile.open(mode= "r:*", fileobj = BytesIO(response.content))
    file_images = list()
    for member in tar.getnames():
        image = np.asarray(bytearray(tar.extractfile(member).read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image,(256,256))
        file_images.append(image)
    all_file_images.append(file_images)

# extracting the single training image from each file as numpy arrays
single_file_images = list()
for filename in train_text_file_list:
    url3 = "https://storage.googleapis.com/uga-dsp/project2/data/"+filename+".tar"
    response = requests.get(url3)
    tar = tarfile.open(mode= "r:*", fileobj = BytesIO(response.content))
    for member in tar.getnames():
        image = np.asarray(bytearray(tar.extractfile(member).read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image,(256,256))
        break
    single_file_images.append(image)

# extracting the training mask from each file as numpy arrays
all_masks = list()
for filename in train_text_file_list:
    url3 = "https://storage.googleapis.com/uga-dsp/project2/masks/"+filename+".png"
    resonse = requests.get(url)
    mask = np.asarray(bytearray(response.content()), dtype="uint8")
    mask = cv2.imdecode(mask, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(256,256))
    # for better visual representation
    mask[mask==0]=0
    mask[mask==1]=128
    mask[mask==2]=255
    all_masks.append(mask)
