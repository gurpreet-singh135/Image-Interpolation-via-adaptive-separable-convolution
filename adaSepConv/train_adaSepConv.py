import tensorflow as tf
from model import *
from prepare_dataset_utils import *
import config as config
import glob

if __name__ == '__main__':

    training_filenames = glob.glob(config.DATASET_DIR+"/*")
    print(training_filenames)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=True)

    model = create_model(config.INPUT_SHAPE)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(clipvalue = 1),
        loss = custom_loss,
        metrics=['accuracy'],
    )
    model.summary()

    model.fit(get_training_dataset(training_filenames), steps_per_epoch=2500*len(training_filenames)/config.BATCH_SIZE, epochs=config.EPOCHS,callbacks = [lr_callback])
    print("saving trained model to current working directory")
    save_model(model)
    print("Model saved successfully please check current working directory")