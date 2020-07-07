import tensorflow as tf
import cv2 
import matplotlib.pyplot as plt
import random
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import gc
import os
import glob
import math
import time
AUTO = tf.data.experimental.AUTOTUNE
import adap_conv_model_config as config
import sys
sys.path.append('../')
from create_dataset_utils import *
from adap_conv_model_utils import *


model=create_model()

#Specify weights file name here
checkpoint_name=config.CHECKPOINT_NAME
path=config.CHECKPOINT_PATH
model.load_weights(glob.glob(path)[0]+checkpoint_name)

#Specify file paths of two frames and interpolated frame path here 
frame1_path=config.FRAME1_PATH
frame1_name=config.FRAME1_NAME
frame2_path=config.FRAME1_PATH
frame2_name=config.FRAME2_NAME
interpolated_frame_path=config.INTERPOLATED_FRAME_PATH
frame1=cv2.imread(frame1_path+frame1_name)
frame2=cv2.imread(frame2_path+frame2_name)
predict_image(model,frame1,frame2,interpolated_frame_path=interpolated_frame_path,save_orignal_frames=False)
print("Success")
