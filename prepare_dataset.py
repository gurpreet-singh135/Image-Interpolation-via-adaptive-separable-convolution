import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import glob
import random
from PIL import Image


#Function to generate an image array when centre pixels are provided
def create_image(frame,i,j):
    if i-75>=0:
        if i+75<=1080-1:
            if j-75>=0:
                if j+75<=1920-1:
                    new_image=frame[i-75:i+75,j-75:j+75,0:3]
                else:
                    new_image=frame[i-75:i+75,1920-1-150:1920-1,0:3]
            else:
                new_image=frame[i-75:i+75,0:150,0:3]
        elif j-75>=0:
                if j+75<=1920-1:
                    new_image=frame[1080-1-150:1080-1,j-75:j+75,0:3]
                else:
                    new_image=frame[1080-1-150:1080-1,1920-1-150:1920-1,0:3]
        else:
            new_image=frame[1080-1-150:1080-1,0:150,0:3]
    elif j-75>=0:
                if j+75<=1920-1:
                    new_image=frame[0:150,j-75:j+75,0:3]
                else:
                    new_image=frame[0:150,1920-1-150:1920-1,0:3]
    else:
        new_image=frame[0:150,0:150,0:3]
    return new_image

#Function to calculate average value optical flow in two images and return indexes of pixels having max individual flow value
def calc_simple_flow(image1,image2):
    flow = cv.optflow.calcOpticalFlowSF(image1, image2, layers=3, averaging_block_size=3, max_flow=4)
    n = np.sum(1 - np.isnan(flow), axis=0)
    n = np.sum(n,axis=0)
    # print(n)
    flow[np.isnan(flow)] = 0
    return np.linalg.norm(np.sum(flow, axis=(0, 1)) / n),np.unravel_index(flow.argmax(), flow.shape)[0:2],np.unravel_index(flow.argmin(), flow.shape)[0:2]

def avg_flow(image1,image2):
    flow = cv.optflow.calcOpticalFlowSF(image1, image2, layers=3, averaging_block_size=3, max_flow=4)
    n = np.sum(1 - np.isnan(flow), axis=0)
    n = np.sum(n,axis=0)
    # print(n)
    flow[np.isnan(flow)] = 0
    return np.linalg.norm(np.sum(flow, axis=(0, 1)) / n)
#Function to generate pixel using max flow and min flow found randomly
def create_random_image_crops_pixels(frame1,frame2,random_number=10):
    max=0
    min=256
    for x in range(random_number):
        i = random.randint(0,1080)
        j=random.randint(0,1920)
        temp_image1=create_image(frame1,i,j)
        temp_image2=create_image(frame2,i,j)
        temp_flow,_,_=calc_simple_flow(temp_image1,temp_image2)
#         print(temp_flow)
        if temp_flow>max:
            i1=i
            j1=j
            max=temp_flow
        if temp_flow<min:
            i2=i
            j2=j
            min=temp_flow
#     print(i,j,max)
    return (i1,j1),(i2,j2)
def create_random_crops_based_on_Prob(frame1,frame2,random_number=20,flow_threshold = 25):
    list_of_points = []
    for x in range(random_number):
        i = random.randint(0,1080)
        j=random.randint(0,1920)
        temp_image1=create_image(frame1,i,j)
        temp_image2=create_image(frame2,i,j)
        flow = avg_flow(temp_image1,temp_image2)
        if random.random() < flow / flow_threshold:
            list_of_points.append((i,j))
            # print(flow)
    return list_of_points

#(i1,j1) has pixel values for max flow and (i2,j2) for least


#Function to generate individual pixel for which flow is max or min
def create_image_crops_pixels(frame1,frame2):
    _,index_high,index_low=calc_simple_flow(frame1,frame2)
    return index_high,index_low

def save_image(frames,pixel_values,type="train",example_number=0):
    image=create_image(frames[0],pixel_values[0],pixel_values[1])
    cv.imwrite("./"+type+"_dataset/"+type+"_"+str(example_number)+"_0.jpg",image)
    image=create_image(frames[1],pixel_values[0],pixel_values[1])
    cv.imwrite("./"+type+"_dataset/"+type+"_"+str(example_number)+"_1.jpg",image)
    image=create_image(frames[2],pixel_values[0],pixel_values[1])
    cv.imwrite("./"+type+"_dataset/"+type+"_"+str(example_number)+"_2.jpg",image)
    return
def save_image2(frames,pixel_values,example_number=0):
    image=create_image(frames[0],pixel_values[0],pixel_values[1])
    cv.imwrite("../dataset/"+str(example_number).zfill(6)+"_0.jpg",image)
    image=create_image(frames[1],pixel_values[0],pixel_values[1])
    cv.imwrite("../dataset/"+str(example_number).zfill(6)+"_1.jpg",image)
    image=create_image(frames[2],pixel_values[0],pixel_values[1])
    cv.imwrite("../dataset/"+str(example_number).zfill(6)+"_2.jpg",image)

def is_jumpcut(frame1, frame2, threshold=np.inf):

    x = np.histogram(frame1.reshape(-1),np.arange(256))[0]
    y = np.histogram(frame2.reshape(-1),np.arange(256))[0]
    res = np.linalg.norm(x-y)
    #use value of threshold approximately 15000
    return res > threshold

def frame_capture_from_videos(list_of_files):
    """
    takes a list of video files from which we
    can read frames and stores them in a directory
    """
    if not os.path.isdir("../training_dataset"):
        os.mkdir("../training_dataset")
    for path in list_of_files:
        vidcap = cv.VideoCapture(path)
        success,image = vidcap.read()
        count = 0
        file_name  = os.path.basename(os.path.splitext(path)[0])
        if not os.path.isdir("../training_dataset/frames_"+ file_name):
            os.mkdir("../training_dataset/frames_"+ file_name)
            while success:

                cv.imwrite("/home/z3u5/Desktop/training_dataset/frames_%s/%s.jpg" % (file_name,str(count).zfill(6)), image)     # save frame as JPEG file      
                success,image = vidcap.read()
                print('Read a new frame: ', success)
                count += 1
###To Test Simple Flow###

# if __name__=="__main__":
#     image1_path = "/home/z3u5/Downloads/DAVIS-2017-Unsupervised-trainval-Full-Resolution/DAVIS/JPEGImages/Full-Resolution/bear/00000.jpg"
#     image2_path = "/home/z3u5/Downloads/DAVIS-2017-Unsupervised-trainval-Full-Resolution/DAVIS/JPEGImages/Full-Resolution/bear/00001.jpg"
#     image1 = np.array(Image.open(image1_path))
#     image2 = np.array(Image.open(image2_path))
#     flow = calc_simple_flow(image1,image2)
#     print(flow)