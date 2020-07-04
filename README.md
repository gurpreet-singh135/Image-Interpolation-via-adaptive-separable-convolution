# Image-Interpolation-via-separable-convolution
Our Project consists of Tensorflow implementation of two papers namely [Video Frame Interpolation via Adaptive Convolution](https://arxiv.org/pdf/1703.07514.pdf) and [Video Frame Interpolation via Adaptive Separable Convolution](https://arxiv.org/pdf/1708.01692.pdf). We have implemented both papers separately and included results after training respective models. We have also included pre-trained models in this repository so you can **double the  FPS of your own videos**, to start please follow steps given below.
## 1. Adaptive Separable Convolution
In this section we include results of [Video Frame Interpolation via Adaptive Separable Convolution](https://arxiv.org/pdf/1708.01692.pdf) by training our own model and applying it on stock videos.
###### Here videos are shown with 0.25x speed (left : original, right : our interpolated result)
[![Watch the video](https://github.com/gurpreet-singh135/Image-Interpolation-via-separable-convolution/blob/master/video.png)](https://vimeo.com/434104472)

To reproduce this result from our pretrained models in our repository, follow the following steps carefully :
1. Please make sure that you have CUDA installed and working on your machine. Also check properly whether Tensorflow GPU is working properly or not otherwise it will take a lot of time for interpolation.
2. Please install OpenCV python for reading and writing frames to videos.
3. After you have verified above installations open terminal and navigate to **adaSepConv** folder in repository using 
```
cd adaSepConv
```
4. Run test_video.py using 
```
python test_video.py
```
5. It will prompt you to provide path of video file for which you want to double FPS. You can provide any video file of any resolution, for example we have included file ```juggle.mp4``` in repository, you can provide its path (Note: don't provide video files with duration more than 20 sec as 20 sec 1080p video takes about 15 min to process)
```enter path to video file : /path/to/video/file.mp4```
6. You can alternatively see results by interpolating frame between two given frames which will take lot less time. To do this run ```test_frame.py``` in terminal it will prompt to provide paths of two frames between which you want to get frame. For easy example just provide path to ```frame1.jpg``` and ```frame3.jpg``` **already in repository** and a file ```interpolated_frame.jpg``` will be generated
```
enter path to first image file : /path/to/first/image.jpg
enter path to second image file : /path/to/second/image.jpg
```
7. You can also train your own model by running ```train.py``` in same directory. To train model according to your specific hyperparameters you can edit ```config.py``` file in same directory to change hyperparameters like BATCH_SIZE etc. Model takes about 10 hours on 1050ti with limited dataset in repository. To include more dataset, just download ```.tfrecord``` files from [drive link](https://drive.google.com/drive/folders/1vGHMMOX7lHZ41lbZxCsgvdm6ZJAvLC_t?usp=sharing) and paste them in ```dataset``` folder in repository and run ```train.py``` 


here's the [drive link](https://drive.google.com/drive/folders/1vGHMMOX7lHZ41lbZxCsgvdm6ZJAvLC_t?usp=sharing) to see our interpolated results and dataset
