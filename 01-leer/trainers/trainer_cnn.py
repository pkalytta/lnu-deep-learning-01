from datasets.flowers_datasets import get_dataset
from models.flower_cnn import get_model

import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np 


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()

class TrainerCNN:

    def __init__(self, epochs, learning_rate, val_split, train_split, **kwargs):
        self.epoch = epochs
        self.learning_rate = learning_rate
        self.val_split = val_split
        self.train_split = train_split

    def train(self, plot=True, image_width = 180, image_height = 180, batch_size = 32, **kwargs):
        training_set, validation_set, num_classes, class_names = get_dataset(
            split = self.train_split, batch_size=batch_size, image_width = image_width, image_height = image_height)

        print(class_names)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./log")

        model = get_model(num_classes, image_width, image_height, self.learning_rate)

        history = model.fit(
            training_set,
            validation_data=validation_set,
            epochs=self.epoch,
            callbacks=[tensorboard_callback]
        )

        # Test prediction
        sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
        sunflower_path = tf.keras.utils.get_file(
            'Red_sunflower', origin=sunflower_url)

        img = tf.keras.preprocessing.image.load_img(
            sunflower_path, target_size=(image_height, image_width)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
                class_names[np.argmax(score)], 100 * np.max(score))
        )

