import numpy as np
import urllib
import cv2
import requests
import tarfile
from io import BytesIO


def get_data(path,train=True, small=True):
    ''' function to get the training and testing data as pyhton list.
    Keyword arguments:
    path -- path to project directory conataining data and mask folder
    train -- flag for traning and testing dataset (default = True)
    small -- flag for subdata set (default = True)
    return:
    X_train, y_train and X_test
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
    print('Getting images ... ')
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
            #image = zero_padding(image)
            video.append(image)
            if train:
                # path for mask
                mask = np.asarray(bytearray(response.content), dtype="uint8")
                mask = cv2.imdecode(mask, cv2.IMREAD_GRAYSCALE)
                #mask = zero_padding(mask)
                mask[mask==0]=0
                mask[mask==1]=0
                mask[mask==2]=1
                video_mask.append(mask)
            i = i + 1
            print(i)
            if(small):
                break
        np_video = np.array(video)
        X.append(np_video)
        np_video = None
        if train:
            np_video_mask = np.array(video_mask)
            y.append(np_video_mask)
            np_video_mask = None
    print('Done!')
    if train:
        return X, y
    else:
        return X

def zero_padding(input_array, final_dimension = [640,640]):
    input_array_size = input_array.shape
    if input_array_size[0] != final_dimension[0] or input_array_size[1] != final_dimension[1]:
        temp = np.zeros((final_dimension[0],final_dimension[1]))
        temp[:input_array_size[0],:input_array_size[1]] = input_array
        return temp
    else:
        return input_array

X_test = get_data("https://storage.googleapis.com/uga-dsp/project2",train=False, small=False)

def calculate_variance(data_set):
    ''' function to calculate variaince over a dataset.
    Keyword arguments:
    data set -- the data set over which variance has to be calculated
    '''
    var = []
    for i in range(len(data_set)):
        variances = np.var(data_set[i],axis=0)
        variances = (variances/np.max(variances))
        var.append(variances)
    return var

## path for the traing file
url = path + "/train.txt"
train_text_file_list = requests.get(url).text
train_text_file_list = train_text_file_list.split()

## Generating and saving the variance dataset
var = np.array(calculate_variance(X_train))
print(var.shape)
i=0
for im in var:
    print(im.shape)
    np.unique(im)
    for x in range(0,im.shape[0]):
        for y in range(0,im.shape[1]):
            if im[x,y] >= 0.5:
                im[x,y] = 255
            else:
                im[x,y] = im[x,y]*255
    np.unique(im)
    cv2.imwrite("/home/ajpanchmia/test_input_images/"+train_text_file_list[i]+".png", im)
    i+=1


-var = []
for i in range(114):
    variances = np.var(X_test[i],axis=0)
    variances = (variances/np.max(variances))
    var.append(variances)


var = np.array(var)
print(var.shape)
i=0
for im in var:
    print(im.shape)
    np.unique(im)
    for x in range(0,im.shape[0]):
        for y in range(0,im.shape[1]):
            if im[x,y] >= 0.5:
                im[x,y] = 255
            else:
                im[x,y] = im[x,y]*255
    np.unique(im)
    cv2.imwrite("/home/ajpanchmia/test_output_images/"+test_text_file_list[i]+".png", im)
    i+=1

var = np.array(var)
print(var.shape)

url2 = "https://storage.googleapis.com/uga-dsp/project2/test.txt"
test_text_file_list = requests.get(url2).text
test_text_file_list = test_text_file_list.split()

i=0
for im in var:
    print(im.shape)
    og_columns = im.shape[1]
    og_rows = im.shape[0]
    image = zero_padding(im)
    print(og_columns,og_rows)
    image = image[np.newaxis,...]
    image = image[...,np.newaxis]
    print(image.shape)
    mask = model.predict(image)
    mask = mask[0,...]
    mask = mask[...,0]
    mask = mask[0:og_rows, 0:og_columns]
    print(np.unique(mask))
    for x in range(0,mask.shape[0]):
        for y in range(0,mask.shape[1]):
            if mask[x,y] >= 0.6:
                mask[x,y] = 2
            else:
                mask[x,y] = 0
    print(mask.shape)
    print(np.unique(mask))
    print(mask.shape)
    cv2.imwrite("/home/ajpanchmia/output_masks_var/"+test_text_file_list[i]+".png", mask)
    print(".......................................",i)
    i += 1
