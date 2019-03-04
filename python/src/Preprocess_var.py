import numpy as np
import urllib
import cv2
import requests
import tarfile
from io import BytesIO


def get_data(path,train=True, small=True):
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
        X = []
        y = []
        list = train_text_file_list
    else:
        X = []
        list = test_text_file_list
    print('Getting and resizing images ... ')
    i = 0
    for filename in list:
        # path for data
        #https://storage.googleapis.com/uga-dsp/project2/data/
        video = []
        video_mask = []
        url3 = path+"/data/"+filename+".tar"
        response_0 = requests.get(url3)
        tar = tarfile.open(mode= "r:*", fileobj = BytesIO(response_0.content))
        url4 = path + "/masks/" + filename+".png"
        response = requests.get(url4)
        for member in tar.getnames():
            image = np.asarray(bytearray(tar.extractfile(member).read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
            #image = cv2.resize(image,(size[0],size[1]))
            video.append(image)
            if train:
                # path for mask
                mask = np.asarray(bytearray(response.content), dtype="uint8")
                mask = cv2.imdecode(mask, cv2.IMREAD_GRAYSCALE)
                #mask = cv2.resize(mask,(size[0],size[1]))
                mask[mask==0]=0
                mask[mask==1]=0
                mask[mask==2]=1
                video_mask.append(mask)
            i = i + 1
            if(small):
                break
        np_video = np.array(video)
        np_video_mask = np.array(video_mask)
        X.append(np_video)
        y.append(np_video_mask)
    print('Done!')
    if train:
        return X, y
    else:
        return X

X_train, y_train = get_data("https://storage.googleapis.com/uga-dsp/project2",train=True, small=False)
