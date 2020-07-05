import os
import numpy as np
import cv2 as cv
import time
import random
from cv2.optflow import calcOpticalFlowSF
from PIL import Image
import glob
from create_dataset_utils import *
import create_dataset_config as config

video_path = config.VIDEO_PATH+"\*"
filenames=glob.glob(video_path)
dataset_path=glob.glob(config.TFRECORD_DATASET_DIR+"\\")


for filename in filenames:
    tfrecords_filename =dataset_path+os.path.basename(filename)[:-4]+'.tfrecord'
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
    os.rename(tfrecords_filename,tfrecords_filename[:-9]+"_"+str(total_patches)+".tfrecord")
print("Success")