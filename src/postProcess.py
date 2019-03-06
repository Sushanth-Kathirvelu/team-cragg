class postProcess:
    def predict(path,var,model,savePath,zeroPadd = True, size = None):

        """function to predict the y_yest
        path -- path of the train.text or file.text
        var -- numpy array for y_test
        model -- loaded model
        savePath -- path forsaving the images
        zeroPadd = flag weather the data set is zero padded or not
        (default=True)"""

        if not os.path.exist(savePath):
            os.mkdir(savePath)
        file = open(path)
        i=0
        for im in var:
            hashName = file[i].strip()
            og_columns = im.shape[1]
            og_rows = im.shape[0]
            if zeroPadd:
                image = zero_padding(im)
            else:
                image = cv2.resize(image,(size[1],size[0]))
            image = image[np.newaxis,...]
            image = image[...,np.newaxis]
            mask = model.predict(image)
            mask = mask[0,...]
            mask = mask[...,0]
            if zeroPadd:
                mask = mask[0:og_rows, 0:og_columns]
            else:
                mask = cv2.resize(mask,(og_columns,og_rows))
            for x in range(0,mask.shape[0]):
                for y in range(0,mask.shape[1]):
                    if mask[x,y] >= 0.6:
                        mask[x,y] = 2
                    else:
                        mask[x,y] = 0
            cv2.imwrite(savePath + "/" + hashName + ".png", mask)
            print(".......................................",i)
            i += 1
