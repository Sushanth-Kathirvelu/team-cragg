import numpy as np
import urllib
import cv2
import requests
import tarfile
from io import BytesIO


def get_data(path, size ,train=True, small=True):
    ''' function to get the training and testing data as np array.
    Keyword arguments:
    path -- path to project directory conataining data and mask folder
    size -- tuple for the new image dimensions
    train -- flag for traning and testing dataset (default = True)
    small -- flag for subdata set (default = True)
    '''
    # train file path
    url = path + "/train.txt"
    train_text_file_list = requests.get(url).text
    train_text_file_list = train_text_file_list.split()
    # test file path
    url2 = path + "/test.txt"
    test_text_file_list = requests.get(url2).text
    test_text_file_list = test_text_file_list.split()
    if train:
        X = np.zeros((211, size[0], size[1], 1), dtype=np.float32)
        y = np.zeros((211, size[0], size[1], 1), dtype=np.float32)
        list = train_text_file_list
    else:
        X = np.zeros((114, im_height, im_width, 1), dtype=np.float32)
        list = test_text_file_list
    print('Getting and resizing images ... ')
    i = 0
    for filename in list:
        # path for data
        #https://storage.googleapis.com/uga-dsp/project2/data/
        url3 = path+"/data/"+filename+".tar"
        response_0 = requests.get(url3)
        tar = tarfile.open(mode= "r:*", fileobj = BytesIO(response_0.content))
        if train:
            url4 = path + "/masks/" + filename+".png"
            response = requests.get(url4)
        for member in tar.getnames():
            image = np.asarray(bytearray(tar.extractfile(member).read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image,(size[0],size[1]))
            X[i, ..., 0] = image / 255
            if train:
                # path for masks
                mask = np.asarray(bytearray(response.content), dtype="uint8")
                mask = cv2.imdecode(mask, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask,(size[0],size[1))
                mask[mask==0]=0
                mask[mask==1]=0
                mask[mask==2]=1
                y[i, ..., 0] = mask
            i = i + 1
            if(small):
                break
    print('Done!')
    if train:
        return X, y
    else:
        return X

X_train, y_train = get_data("https://storage.googleapis.com/uga-dsp/project2",(256,256) ,train=True, small=True)
