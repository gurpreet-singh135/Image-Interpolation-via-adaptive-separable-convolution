# The size of the input images to be fed to the network during training.
CROP_SIZE_fraction: float = 128/150

# The size of the patches to be extracted from the datasets
PATCH_SIZE = (150, 150)

# Number of epochs used for training
EPOCHS: int = 10

# Kernel size of the custom Separable Convolution layer
OUTPUT_1D_KERNEL_SIZE: int = 51

# The batch size used for mini batch gradient descent
BATCH_SIZE: int = 1

# Path to the dataset directory
TFRECORD_DATASET_DIR = '../dataset'

#input shape
INPUT_SHAPE = (128,128,6)

#Dataset directory
DATASET_DIR = '../dataset'

#Prediction height 
PREDICTION_H: int = 128

#Prediction weight
PREDICTION_W: int = 128

#Prediction Batch
PREDICTION_BATCH: int = 1