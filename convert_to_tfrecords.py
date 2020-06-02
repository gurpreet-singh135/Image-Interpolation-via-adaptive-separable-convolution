import numpy as np
from PIL import Image
from matplotlib import image
import matplotlib.pyplot as plt
import os
import IPython.display as display
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_to_tfrecord(tfrecord_filepath,images_path):

    images_path = sorted(images_path)
    writer = tf.io.TFRecordWriter(tfrecord_filepath)
    for i in range(len(images_path)-2):
        img_paths = images_path[i:i+3]
        
        img_raw1 = open(img_paths[0],'rb').read()
        img_raw2 = open(img_paths[1],'rb').read()
        img_raw3 = open(img_paths[2],'rb').read()
        rows = np.array(Image.open(img_paths[0])).shape[0]
        cols = np.array(Image.open(img_paths[0])).shape[1]
    #     annotation_raw = annotation.tostring()
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'img1': _bytes_feature(img_raw1),
            'img2': _bytes_feature(img_raw2),
            'img3': _bytes_feature(img_raw3),
            'height': _int64_feature(rows),
            'width': _int64_feature(cols)}))

        writer.write(example.SerializeToString())

    writer.close()
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [1080,1920, 3]) # explicit size needed for TPU
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
    img1 = decode_image(example['img1'])
    img2 = decode_image(example['img2'])
    img3 = decode_image(example['img3'])
    height = tf.cast(example['height'], tf.int32)
    width = tf.cast(example['width'], tf.int32)
    return img1,img2,img3,height,width

