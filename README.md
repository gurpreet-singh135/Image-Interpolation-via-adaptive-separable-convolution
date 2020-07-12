# Image-Interpolation-via-adaptive-separable-convolution

## Overview

Frame Interpolation consists of adding an intermediate frame between all pairs of consecutive frames in a video to raise the overall fps of the video. Our Project consists of Tensorflow implementation of two papers namely [Video Frame Interpolation via Adaptive Convolution](https://arxiv.org/pdf/1703.07514.pdf) and [Video Frame Interpolation via Adaptive Separable Convolution](https://arxiv.org/pdf/1708.01692.pdf). We have implemented both papers separately and included results after training respective models. We have also included pre-trained models in this repository so you can **double the  FPS of your own videos**, to start please follow steps given below.
## 1. Results
 ### Adaptive Convolution Model Results

 Interpolated Video:

 [![Adaptive Convolution Model Video Result](https://github.com/gurpreet-singh135/Image-Interpolation-via-separable-convolution/blob/master/video.png)](https://vimeo.com/434104472)
 
 ### Adaptive Separable Convolution Model Results
 Interpolated Video :
 ###### Here videos are shown with 0.25x speed (left : original, right : our interpolated result)
 [![Adaptive Separable Convolution Model Video Result](https://github.com/gurpreet-singh135/Image-Interpolation-via-separable-convolution/blob/master/video.png)](https://vimeo.com/434104472)
## 2. Install Dependencies
As this model is built on python it is recommended to use [Anaconda](https://www.anaconda.com/) to install all the dependencies for easy installations.
###  Tensorflow
For non-gpu users: 

>Note:
> If you don't have a GPU better than 1050Ti then you should probably use google colaboratory. You will not need to install any of the dependency too.
```
pip install tensorflow
``` 
For gpu users:
``` 
pip install tensorflow-gpu
``` 
>Note:
>Install required version of cudatoolkit so that tf-gpu can work
###  OpenCV
``` 
pip install opencv-contrib
``` 
###  Skikit-learn (For Adaptive Convolution Model only)
``` 
pip install scikit-learn
``` 

## 3. Create dataset
If you don't want to create your own dataset and use ours, follow this google [drive link](https://drive.google.com/drive/folders/1vGHMMOX7lHZ41lbZxCsgvdm6ZJAvLC_t?usp=sharing) and use all the files with extension .tfrecord in it.
Our dataset contains nearly 70,000 * 3 ( Three frames per data ) image patches as of now.

To create your own dataset, open [create_dataset_config.py](https://github.com/gurpreet-singh135/Image-Interpolation-via-adaptive-separable-convolution/blob/master/create_dataset_config.py) and specify the path to the folder containing your videos to be used as dataset and the folder where you want to save your tfrecord files.

Now run:
```
python create_dataset_from_video.py
```
This will take time depending on the total length of all of your videos, so sit back and relax.

## 4. Train your model
**If you don't want to train your own model and use our pre-trained models then you can move to step 4.**

First, choose the model you want to use.
### Train Adaptive Convolution Model
Go to AdapConv folder and open [adap_conv_model_utils.py](https://github.com/gurpreet-singh135/Image-Interpolation-via-adaptive-separable-convolution/blob/master/AdapConv/adap_conv_model_utils.py) and specify paths for:
```
TFRECORD_DATASET_DIR - Path to the folder containing your tfrecord files
CHECKPOINT_PATH - Path to the folder you want to save your model weights.
CHECKPOINT_NAME - Name of file containing your model weights.
 ```
 We trained our model for 25 epochs and got the following graphs.
 
 ![Graphs](https://github.com/gurpreet-singh135/Image-Interpolation-via-separable-convolution/blob/master/adap_conv_model_curves.png)


  It can take upto 2 minutes per epoch to be trained in this model on google colab.
 ### Train Adaptive Separable Convolution Model
You can also train your own Adaptive Separable Convolution model by running ```train_adaSepConv.py``` in same directory. To train model according to your specific hyperparameters you can edit ```config.py``` file in same directory to change hyperparameters like BATCH_SIZE etc.To include more dataset, just download ```.tfrecord``` files from [drive link](https://drive.google.com/drive/folders/1vGHMMOX7lHZ41lbZxCsgvdm6ZJAvLC_t?usp=sharing) and paste them in ```dataset``` folder in repository and run ```train_adaSepConv.py``` 

We trained our model for 20 epochs and got following graphs
![Graphs](https://github.com/gurpreet-singh135/Image-Interpolation-via-separable-convolution/blob/master/adaSepConv_model_curves.png)

To train on full dataset it takes about 40 min per epoch on google colab.
 ## 5. Predict Image/Video
 ### Adaptive Convolution Model
 First open [adap_conv_model_config.py](https://github.com/gurpreet-singh135/Image-Interpolation-via-adaptive-separable-convolution/blob/master/AdapConv/adap_conv_model_config.py) and specify paths and names for the following:
 ```
 VIDEO_PATH - Path of the folder containing the video to be interpolated
VIDEO_NAME - Name of the video
INTERPOLATED_VIDEO_PATH - Path to the folder where interpolated video is to be saved
CHECKPOINT_PATH - Path to the folder containing your model weights.
CHECKPOINT_NAME - Name of file containing your model weights.
FRAME1_PATH and FRAME2_PATH - Paths of folders containing frames to be interpolated.
FRAME1_NAME and FRAME2_NAME - Names of the frames
INTERPOLATED_FRAME_PATH - Path to the folder where interpolated frame is to be saved
 ```
 >If you want to use our weights for this model then download them from [here](https://drive.google.com/file/d/1An6kuqsJdA64IDIPXgfz85bwAq1YsHNs/view?usp=sharing) and paste them to AdapConv/pre-trained-models
 
 Go to AdapConv directory
 ```
 cd AdapConv
 ```
To interpolate image run -
 ```
 python predict_image_adap_conv_model.py
 ```
 To interpolate video run -
 ```
 python predict_video_adap_conv_model.py
 ```
It takes around 5 minutes to predict one 1080p resolution frame on google colab in this model. However, this time reduces significantly for lower resolution frames. **If you want a faster prediction go for adaptive separable convolution model**.
 ### Adaptive Seperable Convolution Model
After you have verified above installations open terminal and navigate to **adaSepConv** folder in repository using 
```
cd adaSepConv
```
Run test_video.py using 
```
python test_video.py
```
It will prompt you to provide path of video file for which you want to double FPS. You can provide any video file of any resolution, for example we have included file ```juggle.mp4``` in repository, you can provide its path (Note: don't provide video files with duration more than 20 sec as 20 sec 1080p video takes about 15 min to process)

```enter path to video file : /path/to/video/file.mp4```

You can alternatively see results by interpolating frame between two given frames which will take lot less time. To do this run ```test_frame.py``` in terminal it will prompt to provide paths of two frames between which you want to get frame. For easy example just provide path to ```frame1.jpg``` and ```frame3.jpg``` **already in repository** and a file ```interpolated_frame.jpg``` will be generated

```
enter path to first image file : /path/to/first/image.jpg
enter path to second image file : /path/to/second/image.jpg
```
##### Note : Adaptive Separable convolution model takes much less time for predictions as compared to Adaptive convolution model as seen from fact that it takes about 5 min to predict one 1080p frame from Adaptive convolution model whereas it takes only 3 sec to predict 1080p frame from Adaptive Convolution model. This is because of fact that Adaptive convolution model requires ```height*width``` number of passes to produce one frame whereas Adaptive Separable convolution model requires only ONE pass.
## 6.  Architecture 
Following images illustrate the architecture of the two models.
### Adaptive Convolution Model
 ![Adaptive Convolution Model Architecture](https://github.com/gurpreet-singh135/Image-Interpolation-via-separable-convolution/blob/master/adap_conv_model_architecture.png)
 
 ### Adaptive Separable Convolution Model
  
  ![Adaptive Separable Convolution Model Architecture](https://github.com/gurpreet-singh135/Image-Interpolation-via-separable-convolution/blob/master/adaSepConv_model_architecture.png)

## 7. Credits
Because of the following papers this repository became possible:
 - [Video Frame Interpolation via Adaptive Convolution](https://arxiv.org/pdf/1703.07514.pdf)
 - [Video Frame Interpolation via Adaptive Separable Convolution](https://arxiv.org/pdf/1708.01692.pdf)
## 8. Areas to contribute
There are several areas in which the code can be further optimized like using less GPU memory during prediction and a faster implementation of adaptive convolution which might be done with the help of shift-n-stitch technique. Also, you can add the tfrecords that you create to make predictions even better!


