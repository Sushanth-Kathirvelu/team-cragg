import numpy as np
import urllib
import cv2
import requests
import tarfile
from io import BytesIO
from math import*
import math


def padding_value(tiling_value,og_value):
    r = og_value % tiling_value
    if r == 0:
        padded_value = og_value
    else:
        padded_value = og_value + tiling_value - r
    return padded_value

def imageTiller(im):
    images = list()
    #im = cv2.resize(im,(find_nearest_power_of_two(im.shape[0]),find_nearest_power_of_two(im.shape[1])))
    imgheight=im.shape[0]
    imgwidth=im.shape[1]
    y1 = 0
    M = 128
    N = 128
    i=0
    for y in range(0,imgheight,M):
        for x in range(0, imgwidth, N):
            y1 = y + M
            x1 = x + N
            tiles = im[y:y+M,x:x+N]
            images.append(tiles)
            #cv2.rectangle(im, (x, y), (x1, y1), (0, 255, 0))
            #cv2.imwrite(os.path.join(path , str(str(x) + '_' + str(y)+".png")),tiles)
            i = i + 1
    return images

def zero_padding(input_array, final_dimension = [640,640]):
    input_array_size = input_array.shape
    if input_array_size[0] != final_dimension[0] or input_array_size[1] != final_dimension[1]:
        temp = np.zeros((final_dimension[0],final_dimension[1]))
        temp[:input_array_size[0],:input_array_size[1]] = input_array
        return temp
    else:
        return input_array


def get_data(path, size, train=True, small=True):
    ''' function to get the training and testing data as np array.
    Keyword arguments:
    path -- path to project directory conataining data and mask folder
    size -- tuple for the tile dimensions
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
    im_height = 128
    im_width = 128
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
        url3 = path+"/data/"+filename+".tar"
        response_0 = requests.get(url3)
        tar = tarfile.open(mode= "r:*", fileobj = BytesIO(response_0.content))
        if train:
            url4 = path + "/masks/" + filename+".png"
            response = requests.get(url4)
        for member in tar.getnames():
            image = np.asarray(bytearray(tar.extractfile(member).read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
            #image = cv2.resize(image,(256,256))
            image = zero_padding(image,final_dimension = [padding_value(128,image.shape[0]),padding_value(128,image.shape[1])])
            print(image.shape, " image")
            images = imageTiller(image)
            print(len(images))
            for im in images:
                X.append(im / 255)
            if train:
                # path for masks
                mask = np.asarray(bytearray(response.content), dtype="uint8")
                mask = cv2.imdecode(mask, cv2.IMREAD_GRAYSCALE)
                mask[mask==0]=0
                mask[mask==1]=0
                mask[mask==2]=1
                mask = zero_padding(mask,final_dimension =[padding_value(128,mask.shape[0]),padding_value(128,mask.shape[1])])
                print(mask.shape," mask")
                masks = imageTiller(mask)
                print(len(masks))
                for mk in masks:
                    y.append(mk)
            i = i + 1
            if(small):
                break
    print('Done!')
    if train:
        X_train = np.zeros((len(X),im_height, im_width, 1), dtype=np.float32)
        y_train = np.zeros((len(y),im_height, im_width, 1), dtype=np.float32)
        for i in range(len(X)):
            X_train[i,...,0] = X[i]
            y_train[i,...,0] = y[i]
        print(X_train.shape)
        return X_train, y_train
    else:
        X_test = np.zeros((len(X),im_height, im_width, 1), dtype=np.float32)
        for i in range(len(X)):
            X_train[i,...,0] = X[i]
            return X_test

X,y = get_data("https://storage.googleapis.com/uga-dsp/project2", train=True, small=True)

def imageTiller(im):
    images = list()
    #im = cv2.resize(im,(find_nearest_power_of_two(im.shape[0]),find_nearest_power_of_two(im.shape[1])))
    imgheight=im.shape[0]
    imgwidth=im.shape[1]
    y1 = 0
    M = 128
    N = 128
    i=0
    for y in range(0,imgheight,M):
        for x in range(0, imgwidth, N):
            y1 = y + M
            x1 = x + N
            tiles = im[y:y+M,x:x+N]
            images.append(tiles)
            #cv2.rectangle(im, (x, y), (x1, y1), (0, 255, 0))
            #cv2.imwrite(os.path.join(path , str(str(x) + '_' + str(y)+".png")),tiles)
            i = i + 1
    return images

def zero_padding(input_array, final_dimension = [640,640]):
    input_array_size = input_array.shape
    if input_array_size[0] != final_dimension[0] or input_array_size[1] != final_dimension[1]:
        temp = np.zeros((final_dimension[0],final_dimension[1]))
        temp[:input_array_size[0],:input_array_size[1]] = input_array
        return temp
    else:
        return input_array
