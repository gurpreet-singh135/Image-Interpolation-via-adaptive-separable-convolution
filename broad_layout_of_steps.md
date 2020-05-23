1. Preparing Dataset
    * Computation of mean optical flow using Simple Flow(opencv)
    * Adding tuples of three consecutive images from dataset having optical flow above threshold
    * making tf records of data for using speed up of google tpu
    * Training, validation, test 
2. Building model
    * Model architecture as given [here](https://github.com/martkartasev/sepconv/blob/master/src/model.py)
3. Making output images
    * Using output 1D kernels from model to make interpolated image , pixel by pixel
    * Applying padding to edge cases
    * Calculation of loss between ground truth and estimated image
4. Training
    * Gonna take days maybe but saving checkpoint after training for a while
    * Visual validation after each epoch so that we can check it is improving as expected