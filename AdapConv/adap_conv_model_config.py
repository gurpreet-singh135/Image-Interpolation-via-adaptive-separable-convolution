# Number of epochs used for training
EPOCHS: int = 15

#Path to the folder containing the video to interpolate
#Check forward and bakcward slash for windows or UNIX and make sure to put ending slash
VIDEO_PATH=".\..\\"

# Name of video to be interpolated
VIDEO_NAME="juggle.mp4"

#Path to save the interpolated video
INTERPOLATED_VIDEO_PATH=".\..\\"

# Path to the dataset directory
TFRECORD_DATASET_DIR = '.\..\dataset\\'

# Model checkpoint path
CHECKPOINT_PATH=".\pre-trained-models\\"

#Model checkpoint name
CHECKPOINT_NAME="model_weights"


# Frame paths and names
FRAME1_PATH=".\..\\"
FRAME2_PATH=".\..\\"
FRAME1_NAME="frame1.jpg"
FRAME2_NAME="frame3.jpg"
INTERPOLATED_FRAME_PATH=".\..\\"
