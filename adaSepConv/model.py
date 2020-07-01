import math, re, os
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
from matplotlib import pyplot as plt
import adaSepConv.config as config
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE




def conv_module(x,filters,conv_filter_size,stride,padding='same'):
    x = tf.keras.layers.Conv2D(filters,conv_filter_size,strides = stride,padding = padding,activation = 'relu')(x)
    x = tf.keras.layers.Conv2D(filters,conv_filter_size,strides = stride,padding = padding,activation = 'relu')(x)
    x = tf.keras.layers.Conv2D(filters,conv_filter_size,strides = stride,padding = padding,activation = 'relu')(x)
    return x

def upsample_module(x,filters,conv_filter_size,stride,upsample_size = (2,2),padding='same'):
    x = tf.keras.layers.UpSampling2D(size = upsample_size,interpolation = 'bilinear')(x)
    x = tf.keras.layers.Conv2D(filters,conv_filter_size,strides = stride,padding = padding,activation = 'relu')(x)
    return x


def generating_kernel(x,kernel_dimension , conv_filter_size, stride, padding, upsample_size):
    x = tf.keras.layers.Conv2D(filters = kernel_dimension,kernel_size = conv_filter_size, strides = stride, padding = padding, activation = 'relu')(x)
    x = tf.keras.layers.Conv2D(filters = kernel_dimension,kernel_size = conv_filter_size, strides = stride, padding = padding, activation = 'relu')(x)
    x = tf.keras.layers.Conv2D(filters = kernel_dimension,kernel_size = conv_filter_size, strides = stride, padding = padding, activation = 'relu')(x)
    x = tf.keras.layers.UpSampling2D(size = upsample_size,interpolation = 'bilinear')(x)
    x = tf.keras.layers.Conv2D(filters = kernel_dimension,kernel_size = conv_filter_size, strides = stride, padding = padding, activation = 'relu')(x)
    return x
 
def custom_loss(y_true, y_pred):

    return tf.norm(tf.norm(y_true-y_pred, ord=1, axis=(1,2)),axis = 1)


def create_model(input_shape = config.INPUT_SHAPE):
    conv_filter_size = (3, 3)
    stride = (1, 1)
    padding = 'same'
    upsample_size = (2,2)
    kernel_dimension = config.OUTPUT_1D_KERNEL_SIZE
    pad_dimension = kernel_dimension//2
    
    x = tf.keras.Input(input_shape)
    x_input = x

    i1 = x[:,:,:,0:3]
    i2 = x[:,:,:,3:6]
    shape = tf.shape(i1)

    i1 = tf.pad(i1,[[0,0],[pad_dimension,pad_dimension],[pad_dimension,pad_dimension],[0,0]])
    i2 = tf.pad(i2,[[0,0],[pad_dimension,pad_dimension],[pad_dimension,pad_dimension],[0,0]])

    AvgPooling = tf.keras.layers.AveragePooling2D()
    
    
    x = conv_module(x,32,conv_filter_size,stride,padding)
    x = AvgPooling(x)
    
    
    x_64 = conv_module(x,64,conv_filter_size,stride,padding)
    x_128 = AvgPooling(x_64)
    
    
    x_128 = conv_module(x_128,128,conv_filter_size,stride,padding)
    x_256 = AvgPooling(x_128)
    
    x_256 = conv_module(x_256,256,conv_filter_size,stride,padding)
    x_512 = AvgPooling(x_256)
    
    x_512 = conv_module(x_512,512,conv_filter_size,stride,padding)
    x = AvgPooling(x_512)
    
    x = conv_module(x,512,conv_filter_size,stride,padding)
    
    
    x = upsample_module(x,512,conv_filter_size,stride,upsample_size,padding)
    x += x_512
    x = conv_module(x,256,conv_filter_size,stride,padding)
    
    x = upsample_module(x,256,conv_filter_size,stride,upsample_size,padding)
    x += x_256
    x = conv_module(x,128,conv_filter_size,stride,padding)

    x = upsample_module(x,128,conv_filter_size,stride,upsample_size,padding)
    x += x_128
    x = conv_module(x,64,conv_filter_size,stride,padding)
    
    x = upsample_module(x,64,conv_filter_size,stride,upsample_size,padding)
    x += x_64

    
    k1h = generating_kernel(x,kernel_dimension , conv_filter_size, stride, padding, upsample_size)
    k1v = generating_kernel(x,kernel_dimension , conv_filter_size, stride, padding, upsample_size)
    k2h = generating_kernel(x,kernel_dimension , conv_filter_size, stride, padding, upsample_size)
    k2v = generating_kernel(x,kernel_dimension , conv_filter_size, stride, padding, upsample_size)


    image_patches1 = tf.reshape(tf.image.extract_patches(i1,sizes = [1,kernel_dimension,kernel_dimension,1],strides = [1,1,1,1],rates = [1,1,1,1],padding = 'VALID'),(-1,shape[1],shape[2],kernel_dimension,kernel_dimension,3))
    output_images1 = tf.reduce_sum(tf.reduce_sum(image_patches1*tf.expand_dims(tf.expand_dims(k1h,-2),-1)*tf.expand_dims(tf.expand_dims(k1v,-1),-1),axis = -2),axis = -2)
    

    image_patches2 = tf.reshape(tf.image.extract_patches(i2,sizes = [1,kernel_dimension,kernel_dimension,1],strides = [1,1,1,1],rates = [1,1,1,1],padding = 'VALID'),(-1,shape[1],shape[2],kernel_dimension,kernel_dimension,3))
    output_images2 = tf.reduce_sum(tf.reduce_sum(image_patches2*tf.expand_dims(tf.expand_dims(k2h,-2),-1)*tf.expand_dims(tf.expand_dims(k2v,-1),-1),axis = -2),axis = -2)
    

    return tf.keras.Model(inputs = x_input, outputs =output_images1 + output_images2 , name = 'IIASC')


def save_model(model):
    model_json = model.to_json()
    with open("trained_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("trained_model.h5")
    print("Saved trained_model to disk")

def load_model(filepath_to_model_json,filepath_to_model_h5):
    json_file = open(filepath_to_model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filepath_to_model_h5)
    print("Loaded model from disk")
    return loaded_model

def lr_scheduler(epoch):
    LR_START = 0.00005
    LR_MAX = 0.0001
    LR_MIN = 0.00005
    LR_RAMPUP_EPOCHS = 3
    LR_SUSTAIN_EPOCHS = 3
    LR_EXP_DECAY = .8
    if epoch< LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch< LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
         lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr

def custom_loss(y_true, y_pred):
 
    return tf.norm(tf.norm(y_true-y_pred, ord=1, axis=(1,2)),axis = 1)

