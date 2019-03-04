# cragg
### Video segmentation of Cilia

The task is to design an algorithm that learns how to segment cilia. Cilia are micro-
scopic hairlike structures that protrude from literally every cell in your body. They beat
in regular, rhythmic patterns to perform myriad tasks, from moving nutrients in to mov-
ing irritants out to amplifying cell-cell signaling pathways to generating calcium fluid
flow in early cell differentiation. Cilia, and their beating patterns, are increasingly being
implicated in a wide variety of syndromes that affected multiple organs.
Connecting ciliary motion with clinical phenotypes is an extremely active area of research.
We’ll try to address a very small slice of it here. 

# Goal: find the cilia.

### Prerequisites

List of requirements and links to install them:

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [Anaconda](https://www.anaconda.com/) - Python Environment virtualization.
- Tensorflow-gpu The easiest way to get tensorflow-gpu installed is to do the following:

      conda create --name tf_gpu tensorflow-gpu 
   This will install tensorflow-gpu with all its dependencies in the "tf_gpu" environment. Now activate the environment you just created execute
   
      conda activate tf_gpu
  - Keras: In the conda environment created above install keras as follows:
  
            conda install keras
    
      
    
- [Apache Spark](https://spark.apache.org/downloads.html)
- [Google Cloud Platform or similar service](https://cloud.google.com/docs/)

### Data
The data itself are grayscale 8-bit images taken with DIC optics of cilia biopsies published in [Automated identification of abnormal respiratory ciliary motion in nasal biopsies](http://stm.sciencemag.org/content/7/299/299ra124). For each video, there are provided 100 subsequent frames, which is roughly equal to about 0.5 seconds of real-time video (the framerate of each video is 200 fps)
The data are all available on 

      GCP: gs://uga-dsp/project2
This link was provided by Prof. Shannon Quinn

### Approach 

### References

### Authors

### License
