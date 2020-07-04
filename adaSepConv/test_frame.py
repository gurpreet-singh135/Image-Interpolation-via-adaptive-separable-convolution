
from video_interpolation_utilities import *
from model import load_model
import config as config
import glob
import os
import numpy as np
from cv2 import imread,imwrite
if __name__ == "__main__":
    loaded_model = load_model('./pre-trained-models/model_adaSepConv.json',"./pre-trained-models/model_adaSepConv.h5")

    path_fist_file = input("enter path to first image file : ")
    path_second_file = input("enter path to second image file : ")

    image1 = imread(path_fist_file)
    image2 = imread(path_second_file)
    shape = image1.shape
    print("Predicting your output please wait .....")
    image_interpolated = interpolate_frame(np.expand_dims(np.concatenate([image1,image2],axis = -1),axis = 0),loaded_model,config.PREDICTION_BATCH,config.PREDICTION_H,config.PREDICTION_W)

    imwrite('interpolated_image.jpg',image_interpolated[0]*255.)

    cut_extra('interpolated_image.jpg',shape[0],shape[1])


    print("Image interpolated successfully please check current directory")