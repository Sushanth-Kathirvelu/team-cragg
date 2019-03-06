
# Video segmentation of Cilia

The task is to design an algorithm that learns how to segment cilia. Cilia are micro-
scopic hairlike structures that protrude from literally every cell in your body. They beat
in regular, rhythmic patterns to perform myriad tasks, from moving nutrients in to mov-
ing irritants out to amplifying cell-cell signaling pathways to generating calcium fluid
flow in early cell differentiation. Cilia, and their beating patterns, are increasingly being
implicated in a wide variety of syndromes that affected multiple organs.
Connecting ciliary motion with clinical phenotypes is an extremely active area of research.
We’ll try to address a very small slice of it here. 

### Goal: find the cilia.

## Prerequisites

List of requirements and links to install them:

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [Anaconda](https://www.anaconda.com/) - Python Environment virtualization.
- Tensorflow-gpu The easiest way to get tensorflow-gpu installed is to do the following:

            conda create --name tf_gpu tensorflow-gpu 
   This will install tensorflow-gpu with all its dependencies in the "tf_gpu" environment. Now activate the environment you just created execute
   
            conda activate tf_gpu
 - Keras: In the conda environment created above install keras as follows:
  
            conda install keras
    
      
- [Google Cloud Platform or similar service](https://cloud.google.com/docs/)

## Data
The data itself are grayscale 8-bit images taken with DIC optics of cilia biopsies published in [Automated identification of abnormal respiratory ciliary motion in nasal biopsies](http://stm.sciencemag.org/content/7/299/299ra124). For each video, there are provided 100 subsequent frames, which is roughly equal to about 0.5 seconds of real-time video (the framerate of each video is 200 fps)
The data are all available on 

      GCP: gs://uga-dsp/project2
This link was provided by Prof. Shannon Quinn
The following are the regions of interest:
• 2 corresponds to cilia 
• 1 corresponds to a cell
• 0 corresponds to background (neither a cell nor cilia)

## Approach 
U-Net: This model is the most popular method for segmentation published by Olaf Ronneberger (https://arxiv.org/abs/1505.04597) et al.
The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization.
The provided model is basically a convolutional auto-encoder, but with a twist - it has skip connections from encoder layers to decoder layers that are on the same "level". See picture below (note that image size and numbers of convolutional filters in this tutorial differs from the original U-Net architecture).

This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

### Inputs

* Resize (256,256)
* Resize (512,512)
* Zero Padding(640,640)
* Zero Padding(640,640),Optical Flow of 2 Images
* Zero Padding(640,640),Optical Flow of Video
* Zero Padding(640,640),Grey Scale Variance
* Zero Padding(640,640),FFT Variance
* Zero Padding(640,640),Mean of Optical Flow and Variance

## Running

            python main.py
  This gives two options to the user
  1) Do Preprocess
  2) Run unet models
  On running 2 it gives 8 different unet combinations as above
  
## References
* Unet (https://arxiv.org/abs/1505.04597)
* Optical Flow (https://opencv-python- tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html)
* MaskRCNN (https://github.com/matterport/Mask_RCNN)
* Thiramisu (https://github.com/fastai/courses/blob/master/deeplearning2/tiramisu-keras.ipynb)
* FFT Variance (https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html)
* Grey Scale Variance (https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html)

## Contibutors
See the contributors file for details. 

[Contributors](https://github.com/dsp-uga/team-cragg/blob/develop/Contributors.md)

## License
This project is licensed under the MIT License- see the [LICENSE.md](https://github.com/dsp-uga/team-cragg/blob/master/LICENSE) file for details
