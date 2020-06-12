import math, re, os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import model_from_json
import cv2
from matplotlib import pyplot as plt

def capture_frames_from_video(list_of_videofiles):
    for path in list_of_videofiles:
        vidcap = cv2.VideoCapture(path)
        success,image = vidcap.read()
        count = 0
        file_name  = os.path.basename(os.path.splitext(path)[0])
        if not os.path.isdir("./frames_"+ file_name):
            os.mkdir("./frames_"+ file_name)
            while success:

                cv2.imwrite("./frames_%s/%s.jpg" % (file_name,str(count).zfill(6)), image)     # save frame as JPEG file      
                success,image = vidcap.read()
                print('Read a new frame: ', success)
                count += 2


def pad_frame(image,pad):
    return np.pad(image, ((pad,pad),(0,0),(0,0)), 'constant')


def interpolate_frame_fullHD(img,loaded_model):
    # img = np.concatenate((pad_frame(image1,36),pad_frame(image3,36)),axis = -1)
#     img = np.expand_dims(img, axis=0)/255.
    interpolated_image = np.zeros((6,1152,1920,3))
    img = img/255.
    # img = np.stack([img for i in range(6)],axis = 0)/255.
    # interpolated_image = np.stack([interpolated_image for i in range(6)],axis = 0)
    for row in range(9):
        for col in range(15):
            interpolated_image[:,row*128:row*128+128,col*128:col*128+128,:] = loaded_model.predict(img[:,row*128:row*128+128,col*128:col*128+128,:],batch_size = 16,use_multiprocessing = True)
    
    return interpolated_image



def create_interpolated_frames(video_frames_filepath,loaded_model):
  batch_of_images = []
  batch_size = 0
  interpolated_filenumber = 1
  video_frames_filepath = sorted(video_frames_filepath)
  for i in range(len(video_frames_filepath)-1):
    image1_path = video_frames_filepath[i]
    image2_path = video_frames_filepath[i+1]
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    img = np.concatenate([pad_frame(image1,36),pad_frame(image2,36)],axis = -1)
    img = np.expand_dims(img,axis = 0)
    if batch_size < 5:
      if type(batch_of_images) == list:
        batch_of_images = img 
        batch_size += 1
      else:
        batch_of_images = np.concatenate([batch_of_images,img],axis = 0)
        batch_size += 1
      continue
    else :
      batch_of_images = np.concatenate([batch_of_images,img],axis = 0)
      batch_size += 1
      interpolated_images = interpolate_frame_fullHD(batch_of_images,loaded_model)
      for j in range(6):
        image_name = './frames_test3/'+str(interpolated_filenumber).zfill(6)+'.jpg'
        cv2.imwrite(image_name,interpolated_images[j,:,:,:]*255.)
        interpolated_filenumber += 2
      batch_size = 0
      batch_of_images = []



def cut_extra_fullHD(video_frames_list):
  x = 1
  for frame in video_frames_list:
    file_name  = os.path.basename(os.path.splitext(frame)[0])
    current_file_to_cut = str(x).zfill(6)
    if(current_file_to_cut==file_name):
      img = cv2.imread(frame)
      img = img[36:1080+36,:,:]
      cv2.imwrite(frame,img)
      x += 2


def convert_frames_to_video(list_of_files,pathOut,fps):

    frame_array = []
    for i in range(len(list_of_files)):
        # filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(list_of_files[i])
        height, width, layers = img.shape
        size = (width,height)
        print(list_of_files[i])
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


