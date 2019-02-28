#zero padding



from PIL import ImageOps, Image
import cv2
import numpy as np



def zero_padding(hash_list, data_path, desired_size):
    
    """
    This function adds zero padding to all images to get them to the desired size. It then 
    makes a numpy array, normalizes it and adds 1 axis so that it is ready to be given as 
    input to the model
    Sample:
    hash_list = '/home/rutu/DSP/train.txt'
    data_path = "/home/rutu/DSP/train_frame0000/"

    """
    train_npy = []
    with open(hash_list, 'r') as f:
        myNames = [line.strip() for line in f]
    for file in myNames:
        img = cv2.imread(data_path+file+".png", cv2.IMREAD_GRAYSCALE)
        
        pil_img = Image.fromarray(img)

        new_size = pil_img.size
        delta_w = desired_size - new_size[0]
        delta_h = desired_size - new_size[1]
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        padded_im = ImageOps.expand(pil_img, padding)
        padded_im = np.array(padded_im)

        train_npy.append(padded_im)
    train_npy  = np.array(train_npy)
    print(train_npy.shape)

    train_npy = train_npy / 255
    train_npy = np.reshape(train_npy, train_npy.shape + (1,))
    print(train_npy.shape)
    return train_npy


zp = zero_padding(hash_list, data_path, 640)
