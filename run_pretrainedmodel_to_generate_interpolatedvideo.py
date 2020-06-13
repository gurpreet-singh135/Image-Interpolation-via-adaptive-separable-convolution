from model import load_model
from video_interpolation_utilities import *
import glob
import numpy as np
from cv2 import imread,imwrite

if __name__=="__main__":
    print("picking pretrained model in repo")
    loaded_model = load_model("./model2.json","./model2.h5")
    print("do you want to interpolate frame between two frames or double fps of whole video")
    choice = input("enter 1 for whole video conversion, 0 for creating interpolated frame between two given frames :  ")
    if choice == '1':
        print("here")
        path_to_videofiles = input("Provide absolute path of video for which you want to double FPS(remember provide only fullHD video of duration not more than 10 sec as if you dont have good GPU it will take time around 20min) :")
        batch = input("enter batch size for conversion (range 1-4, don't give too large size if you don't have GPU memory greater than 6GB) : ")
        capture_frames_from_video(path_to_videofiles)
        video_filename  = os.path.basename(os.path.splitext(path_to_videofiles)[0])
        video_frames_filepath = glob.glob('./frames_'+ video_filename+'/*jpg')
        print(batch)
        create_interpolated_frames(video_frames_filepath,loaded_model,int(batch),video_filename)
        cut_extra_fullHD(glob.glob(video_frames_filepath))
        fps = input("enter fps of your given video(output video will have double of this value) : ")
        convert_frames_to_video(glob.glob(video_frames_filepath),'.',2*fps)
        print("video converted!")
    if choice == '0':
        path_fist_file = input("enter path to first fullHD file : ")
        path_second_file = input("enter path to second fullHD file : ")
        image1 = imread(path_fist_file)
        image2 = imread(path_second_file)
        img = np.expand_dims(np.concatenate([pad_frame(image1,36),pad_frame(image2,36)],axis = -1),0)
        print("picking pretrained model in repo")
        loaded_model = load_model("./model2.json","./model2.h5")
        interpolated_image = interpolate_frame_fullHD(img,loaded_model,1)
        imwrite('interpolated_image.jpg',interpolated_image)
        print("image interpolated successfully! check in current workind directory")