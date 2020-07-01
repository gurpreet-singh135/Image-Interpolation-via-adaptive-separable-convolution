import tensorflow as tf
AUTO = tf.data.experimental.AUTOTUNE
from create_dataset_utils import *
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
def transform_dataset(img1,img2,img3,height,width):
  center_x=random.randint(39,111)
  center_y=random.randint(39,111)
  img1=img1[center_x-39:center_x+39+1,center_y-39:center_y+39+1,:]
  img3=img3[center_x-39:center_x+39+1,center_y-39:center_y+39+1,:]
  img1=tf.concat([img1,img3],-1)
  return img1,img2[center_x,center_y,:]
def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord , num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset
def get_training_dataset(filenames):
    dataset = load_dataset(filenames, labeled=True)
    dataset=dataset.map(transform_dataset)
    dataset = dataset.cache()
    # dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(128)
    # dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset
def predict_frame(model,frame1,frame2,height,width):
    Frame1=tf.pad(frame1,[[39,39],[39,39],[0,0]])
    Frame2=tf.pad(frame2,[[39,39],[39,39],[0,0]])
    Frame3=np.concatenate((Frame1,Frame2),axis=-1)
    prediction=np.empty((height,width,3),dtype="uint8")
    w=30
    j=0
    print(height,width)
    while j<width-w:
      print(j)
      prediction[:,j:j+w,:]=np.reshape(model.predict(image.extract_patches_2d(Frame3[:,j:j+w-1+79,:],(79,79)),batch_size=128,use_multiprocessing=True),(height,w,3))
      j=j+w
    if j!=width:
        w=width-j
        prediction[:,j:j+w,:]=np.reshape(model.predict(image.extract_patches_2d(Frame3[:,j:j+w-1+79,:],(79,79)),batch_size=128,use_multiprocessing=True),(height,w,3))
    return prediction
class myLoss(tf.keras.losses.Loss):
  def call(self, y_true, y_pred):
    loss=y_true-y_pred
    loss=tf.keras.backend.abs(loss)
    loss=tf.math.reduce_sum(loss)
    return loss
def create_model():
    inputs=tf.keras.layers.Input(shape=[79,79,6])
    x=tf.keras.layers.BatchNormalization()(inputs)
    x=(tf.keras.layers.Conv2D(kernel_size=7, filters=32, padding='valid', activation='relu')(inputs))
    x=(tf.keras.layers.BatchNormalization()(x))
    x=(tf.keras.layers.Conv2D(kernel_size=2, filters=32, padding='valid',strides=(2,2), activation='relu')(x))
    x=(tf.keras.layers.Conv2D(kernel_size=5, filters=64, padding='valid', activation='relu')(x))
    x=(tf.keras.layers.BatchNormalization()(x))
    x=(tf.keras.layers.Conv2D(kernel_size=2, filters=64, padding='valid',strides=(2,2), activation='relu')(x))
    x=(tf.keras.layers.Conv2D(kernel_size=5, filters=128, padding='valid', activation='relu')(x))
    x=(tf.keras.layers.BatchNormalization()(x))
    x=(tf.keras.layers.Conv2D(kernel_size=2, filters=128, padding='valid',strides=(2,2), activation='relu')(x))
    x=(tf.keras.layers.Conv2D(kernel_size=3, filters=256, padding='valid', activation='relu')(x))
    x=(tf.keras.layers.BatchNormalization()(x))
    x=(tf.keras.layers.Conv2D(kernel_size=4, filters=2048, padding='valid', activation='relu')(x))
    x=(tf.keras.layers.Conv2D(kernel_size=1, filters=3362, padding='valid', activation='softmax')(x))
    x=tf.keras.layers.Reshape((41,82,1))(x)
    image_tensors=tf.split(inputs[:,19:60,19:60,:],num_or_size_splits=2,axis=-1)
    kernel_tensors=tf.split(x,num_or_size_splits=2,axis=-2)
    img1=image_tensors[0][:,:,:,0:3]
    img2=image_tensors[1][:,:,:,0:3]
    output1=img1*kernel_tensors[0]
    output2=img2*kernel_tensors[1]
    pixel1=tf.math.reduce_sum(output1,axis=[1,2],keepdims=True)
    pixel2=tf.math.reduce_sum(output2,axis=[1,2],keepdims=True)
    pixel=tf.squeeze(pixel1,axis=[1,2])+tf.squeeze(pixel2,axis=[1,2])
    outputs=pixel
    return tf.keras.models.Model(inputs=inputs,outputs=outputs,name='model')
def predict_image(model,frame1,frame2,interpolated_frame_path,save_orignal_frames=True):
    interpolated_frame=predict_frame(model,frame1,frame2,frame1.shape[0],frame1.shape[1])
    cv2.imwrite(interpolated_frame_path+"interpolated_frame.jpg",interpolated_frame)
    if save_orignal_frames:
        cv2.imwrite(interpolated_frame_path+"frame1.jpg",frame1)
        cv2.imwrite(interpolated_frame_path+"frame2.jpg",frame2)
    return
# Save video feaure not implemented
def predict_video(model,cap,video,save_orignal_video=True):
    ret,frame1=cap.read()
    if ret:
        video.write(frame1)
    ret,frame2=cap.read()
    while ret:
        interpolated_frame=predict_frame(model,frame1,frame2,frame1.shape[0],frame1.shape[1])
        video.write(interpolated_frame)
        video.write(frame2)
        frame1=frame2
        ret,frame2=cap.read()
    return
        
