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
import sys
from adap_conv_model_utils import *
import adap_conv_model_config as config
sys.path.append('../')
from create_dataset_utils import *
#Specify path to tfrecord folder here containing tfrecord files
data_path = config.TFRECORD_DATASET_DIR+"*.tfrecord"
filenames=glob.glob(data_path)
dataset = get_training_dataset(filenames=filenames)

model=create_model()
model.compile(
  optimizer="adamax",
  loss= myLoss(),
  metrics=['accuracy'])
model.summary()

keras.utils.plot_model(model,to_file='adap_conv_model.png',show_shapes=True,show_layer_names=True)


history=model.fit(dataset,epochs=config.EPOCHS,use_multiprocessing=True)

#Specify weights file name here
checkpoint_name=config.CHECKPOINT_NAME
model.save_weights(glob.glob(config.CHECKPOINT_PATH)[0]+checkpoint_name)


print("Model trained and saved as "+checkpoint_name)