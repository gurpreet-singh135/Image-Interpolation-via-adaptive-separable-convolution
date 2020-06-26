#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import cv2 as cv
import time
import random
from cv2.optflow import calcOpticalFlowSF
from PIL import Image
# import IPython.display as display
# from prepare_dataset import *
import glob
from create_dataset_utils import *
AUTO = tf.data.experimental.AUTOTUNE


# In[2]:


# Specify the video path folder and path to store tfrecords
video_path = "F:/DL Frame Interpolation/paaji/video/*.mp4"
filenames=glob.glob(video_path)
dataset_path="F:/DL Frame Interpolation/paaji/dataset/"


# In[3]:





# In[5]:


for filename in filenames:
    tfrecords_filename =dataset_path+os.path.basename(filename)[:-4]+'.tfrecords'
    writer = tf.io.TFRecordWriter(tfrecords_filename)
    video=cv.VideoCapture(filename)    
    total = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    frame_numbers=random.sample(range(total-3), (int)(total/10))
    frame_numbers.sort()
    total_patches=0
    for frame_number in frame_numbers:
        video.set(1,frame_number)
        _,frame0=video.read()
        _,frame1=video.read()
        _,frame2=video.read()
        frames=(frame0,frame1,frame2)
        if not is_jumpcut(frame0,frame2,threshold = 100000):
            total_patches = create_random_crops_based_on_Prob(frames=frames,writer=writer,total_patches=total_patches)
    writer.close()
    os.rename(tfrecords_filename,tfrecords_filename[:-10]+"_"+str(total_patches)+".tfrecords")


# In[ ]:





# In[6]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:





# In[8]:





# In[10]:





# In[ ]:




