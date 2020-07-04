
from video_interpolation_utilities import *
from model import load_model
import config as config
import glob
import os
import numpy as np
from cv2 import imread,imwrite
import shutil
if __name__ == "__main__":
    loaded_model = load_model('./pre-trained-models/model_adaSepConv.json',"./pre-trained-models/model_adaSepConv.h5")

    videofile = input("enter path to video file : ")

    print("Writing video frames in folder  .....")
    fps = capture_frames_from_video(videofile)
    print("Written all frames successfully ...")
    print("Interpolating frames please wait ...")
    video_filename  = os.path.basename(os.path.splitext(videofile)[0])
    print(video_filename)
    create_interpolated_frames(glob.glob("./frames_"+ video_filename+"/*"),loaded_model,config.PREDICTION_BATCH,video_filename)
    
    convert_frames_to_video(sorted(glob.glob("./frames_"+ video_filename+"/*")),'./'+video_filename+'.avi',fps*2)
    shutil.rmtree("./frames_"+ video_filename)
    print("Video generated with double FPS  successfully please check current directory")