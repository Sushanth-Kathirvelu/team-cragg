from UNet import*
class loadModel:
    def load(size,modelName):
        """function to load the images
        size -- tuple for the image dimensions required by the model
        modelName -- name for the model
        """
        if not os.path.exist(savePath):
        model = UNet((size[0],size[1],1))
        model.load_weights(modelName)
        return model
