import numpy as np
from PIL import Image
from matplotlib import image
import matplotlib.pyplot as plt
import os
import IPython.display as display
import tensorflow as tf
import cv2 as cv
import glob
import random

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_to_tfrecord(frame1,frame2,frame3,writer):

    rows = frame1.shape[0]
    cols = frame1.shape[1]
    example = tf.train.Example(features=tf.train.Features(feature={
        'img1': _bytes_feature(tf.image.encode_jpeg(frame1).numpy()),
        'img2': _bytes_feature(tf.image.encode_jpeg(frame2).numpy()),
        'img3': _bytes_feature(tf.image.encode_jpeg(frame3).numpy()),
        'height': _int64_feature(rows),
        'width': _int64_feature(cols)}))
    writer.write(example.SerializeToString())
#     return writer
def decode_image(image_data,height,width):
    image = tf.io.decode_jpeg(image_data, out_type="uint8")
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [height,width, -1]) # explicit size needed for TPU
    return image
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "img1": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "img2": tf.io.FixedLenFeature([], tf.string),
        "img3": tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64)
        # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    height = tf.cast(example['height'], tf.int32)
    width = tf.cast(example['width'], tf.int32)
    img1 = decode_image(example['img1'],height,width)
    img2 = decode_image(example['img2'],height,width)
    img3 = decode_image(example['img3'],height,width)

    return img1,img2,img3,height,width

#Function to generate an image array when centre pixels are provided
def create_image(frame,i,j):
    new_image=frame[i-75:i+75,j-75:j+75,0:3]
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
def create_random_crops_based_on_Prob(frames,writer,total_patches,random_number=20,flow_threshold = 25):
    for x in range(random_number):
        i = random.randint(75,frames[0].shape[0]-76)
        j=random.randint(75,frames[0].shape[1]-76)
        temp_image1=create_image(frames[0],i,j)
        temp_image3=create_image(frames[2],i,j)
        flow = avg_flow(temp_image1,temp_image3)
        if random.random() < flow / flow_threshold:
            temp_image2=create_image(frames[1],i,j)
            total_patches=total_patches+1
            write_to_tfrecord(temp_image1,temp_image2,temp_image3,writer)
    return total_patches

#(i1,j1) has pixel values for max flow and (i2,j2) for least


#Function to generate individual pixel for which flow is max or min
def create_image_crops_pixels(frame1,frame2):
    _,index_high,index_low=calc_simple_flow(frame1,frame2)
    return index_high,index_low


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
def prepare_dataset(img1,img2,img3,height,width):
    fraction = 0.8533333
    img1 = tf.image.central_crop(img1, central_fraction = fraction)
    img3 = tf.image.central_crop(img3, central_fraction = fraction)
    img2 = tf.image.central_crop(img2, central_fraction = fraction)
    return tf.concat([img1,img3],axis = -1),img2