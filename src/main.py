#imports
from PreProcess/extractTar import *
from PreProcess/generateNumpy import *
from PreProcess/generateVideo import *
from PreProcess/loadModel import *
from PreProcess/OFVarMean import *
from PreProcess/opticalFlow import *
from PreProcess/variance import *
from PreProcess/zeroPad import *
from PreProcess/fftVaraiance import *
from PreProcess/resize import *
from postProcess import *
from UNet import *
import unet

#get Input from user
userPath = input("Enter the path where you have the dataset")

#Paths
testFileHashPath = userPath + "/" +"test.txt"
trainFileHashPath = userPath + "/" +"train.txt"
trainMaskPath = userPath + "/" +"masks"
dataPath = userPath + "/" +"data"
singleImgExtractedTainPath =  userPath + "/" +"singleImgExtractedTrain"
singleImgExtractedTestPath =  userPath + "/" +"singleImgExtractedTest"
allImgExtractedTrainPath= userPath + "/" +"allImgExtractedTrain" +"/" +"data"
allImgExtractedTestPath= userPath + "/" +"allImgExtractedTest" +"/" +"data"
resizedImgTrainPath = userPath + "/" +"resizedImgTrain"
resizedImgTestPath = userPath + "/" +"resizedImgTest"
ZeroPaddedImgTrainPath = userPath + "/" +"ZeroPaddedImgTrain"
ZeroPaddedImgTestPath = userPath + "/" +"ZeroPaddedImgTest"
VideoPathTrain = userPath + "/" +"videoTrain"
VideoPathTest = userPath + "/" +"videoTest"
opticalFlow2ImgTrainPath = userPath + "/" +"opticalFlow2ImgTrain"
opticalFlow2ImgTestPath = userPath + "/" +"opticalFlow2ImgTest"
opticalFlowVideoTrainPath = userPath + "/" +"opticalFlowVideoTrain"
opticalFlowVideoTestPath = userPath + "/" +"opticalFlowVideoTest"
greyscaleVarianceImgTrainPath = userPath + "/" +"greyscaleVarianceImgTrain"
greyscaleVarianceImgTestPath = userPath + "/" +"greyscaleVarianceImgTest"
FFTVarianceImgTrainPath = userPath + "/" +"FFTVarianceImgTrain"
FFTVarianceImgTestPath = userPath + "/" +"FFTVarianceImgTest"
OFVarMeanImgTrainPath = userPath + "/" +"OFVarMeanImgTrain"
OFVarMeanImgTestPath = userPath + "/" +"OFVarMeanImgTest"
outputPath = userPath + "/" +"output"
Print("Is this the First Time you run the code? If yes, run Preprocess")
Print("1. Preprocess")
Print("2. UNET")

firstChoice= input("Enter Your Choice : ")

if(firstChoice == 1):
    Print("We are generating all the datas...")
    Print("This May Take Some time...")
    Print("Please Wait...")

    #Creating Data Folders
    extractSingleFile(trainFileHashPath,singleImgExtractedTainPath)
    extractSingleFile(testFileHashPath,singleImgExtractedTestPath)
    extractTar(testFileHashPath,allImgExtractedTrainPath)
    extractTar(trainFileHashPath,allImgExtractedTestPath)
    generateVideo(testFileHashPath,allImgExtractedTrainPath,VideoPathTest)
    generateVideo(trainFileHashPath,allImgExtractedTestPath,VideoPathTrain)
    resize(testFileHashPath,[512,512],singleImgExtractedTestPath,resizedImgTestPath)
    resize(trainFileHashPath,[512,512],singleImgExtractedTestPath,resizedImgTrainPath)
    opticalFlowVideo(testFileHashPath,VideoPathTest,opticalFlowVideoTestPath)
    opticalFlowVideo(trainFileHashPath,VideoPathTrain,opticalFlowVideoTrainPath)
    opticalFlowTwoImg(testFileHashPath,allImgExtractedTestPath,opticalFlowTwoImgTestPath)
    opticalFlowTwoImg(trainFileHashPath,allImgExtractedTrainPath,opticalFlowTwoImgTrainPath)
    calculateGrayScaleVaraiance(testFileHashPath,generateNumpy(trainFileHashPath,testFileHashPath,allImgExtractedTrainPath,trainMaskPath,allImgExtractedTestPath,True),greyscaleVarianceImgTestPath)
    calculateGrayScaleVaraiance(trainFileHashPath,generateNumpy(trainFileHashPath,testFileHashPath,allImgExtractedTrainPath,trainMaskPath,allImgExtractedTestPath,True),greyscaleVarianceImgTrainPath)
    fftVaraiance(testFileHashPath,generateNumpy(trainFileHashPath,testFileHashPath,allImgExtractedTrainPath,trainMaskPath,allImgExtractedTestPath,True),FFTVarianceImgTestPath)
    fftVaraiance(trainFileHashPath,generateNumpy(trainFileHashPath,testFileHashPath,allImgExtractedTrainPath,trainMaskPath,allImgExtractedTestPath,True),FFTVarianceImgTrainPath)
    OFVarMean(testFileHashPath,greyscaleVarianceImgTestPath,opticalFlowVideoTestPath,opticalFlowTwoImgTestPath,OFVarMeanImgTestPath)
    OFVarMean(trainFileHashPath,greyscaleVarianceImgTestPath,opticalFlowVideoTestPath,opticalFlowTwoImgTestPath,OFVarMeanImgTrainPath)

if(firstChoice == 2):
    #Switch to Initiate the required UNET loadModel
    Print("Which UNET Model you want to run?")
    print(" 1: Resize(256,256)")
    print("2. Resize(512,512)")
    print("3. Zero Padding(640,640)")
    print("4. Zero Padding(640,640),Optical Flow of 2 Images")
    print("5. Zero Padding(640,640),Optical Flow of Video")
    print("6. Zero Padding(640,640),Grey Scale Variance")
    print("7. Zero Padding(640,640),FFT Variance")
    print("8.Zero Padding(640,640),Mean of Optical Flow and Variance")
    unetChoice= input("Enter Your Choice : ")
    if(unetChoice == 1):
        train((256,256),resizedImgTrainPath,trainMaskPath)
        model = load((256,256),"unet.hdf5")
        predict(resizedImgTestPath,generateNumpy(trainFileHashPath,testFileHashPath,allImgExtractedTrainPath,trainMaskPath,allImgExtractedTestPath,True),model,outputPath,False,(256,256))
    if(unetChoice == 2):
        train((512,512),resizedImgTrainPath,trainMaskPath)
        model = load((512,512),"unet.hdf5")
        predict(resizedImgTestPath,generateNumpy(trainFileHashPath,testFileHashPath,allImgExtractedTrainPath,trainMaskPath,allImgExtractedTestPath,True),model,outputPath,False,(512,512))
    if(unetChoice == 3):
        train((640,640),ZeroPaddedImgTrainPath,trainMaskPath)
        model = load((640,640),"unet.hdf5")
        predict(ZeroPaddedImgTestPath,generateNumpy(trainFileHashPath,testFileHashPath,allImgExtractedTrainPath,trainMaskPath,allImgExtractedTestPath,True),model,outputPath,False,(640,640))
    if(unetChoice == 4):
        train((640,640),opticalFlow2ImgTrainPath,trainMaskPath)
        model = load((640,640),"unet.hdf5")
        predict(opticalFlow2ImgTestPath,generateNumpy(trainFileHashPath,testFileHashPath,allImgExtractedTrainPath,trainMaskPath,allImgExtractedTestPath,True),model,outputPath,False,(640,640))
    if(unetChoice == 5):
        train((640,640),opticalFlowVideoTrainPath,trainMaskPath)
        model = load((640,640),"unet.hdf5")
        predict(opticalFlowVideoTestPath,generateNumpy(trainFileHashPath,testFileHashPath,allImgExtractedTrainPath,trainMaskPath,allImgExtractedTestPath,True),model,outputPath,False,(640,640))
    if(unetChoice == 6):
        train((640,640),greyscaleVarianceImgTrainPath,trainMaskPath)
        model = load((640,640),"unet.hdf5")
        predict(greyscaleVarianceImgTestPath,generateNumpy(trainFileHashPath,testFileHashPath,allImgExtractedTrainPath,trainMaskPath,allImgExtractedTestPath,True),model,outputPath,False,(640,640))
    if(unetChoice == 7):
        train((640,640),FFTVarianceImgTrainPath,trainMaskPath)
        model = load((640,640),"unet.hdf5")
        predict(FFTVarianceImgTestPath,generateNumpy(trainFileHashPath,testFileHashPath,allImgExtractedTrainPath,trainMaskPath,allImgExtractedTestPath,True),model,outputPath,False,(640,640))
    if(unetChoice == 8):
        train((640,640),OFVarMeanImgTrainPath,trainMaskPath)
        model = load((640,640),"unet.hdf5")
        predict(OFVarMeanImgTestPath,generateNumpy(trainFileHashPath,testFileHashPath,allImgExtractedTrainPath,trainMaskPath,allImgExtractedTestPath,True),model,outputPath,False,(640,640))
