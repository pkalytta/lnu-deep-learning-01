from datasets.flowers_dataset import get_dataset
from models.flower_cnn_1 import get_model
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
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.val_split = val_split
        self.train_split = train_split

    def train(self, plot=True, image_width=180, image_height=180, batch_size=32, **kwargs):
        # get dataset
        training_set, validation_set, num_classes, class_names = get_dataset(split=self.train_split,
                                                                image_width=image_width,
                                                                image_height=image_height,
                                                                batch_size=batch_size)
       
        print(class_names)

        # tensorboard callback to log model training
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./log")

        print(tensorboard_callback)

        # get model
        model = get_model(num_classes=num_classes, lr=self.learning_rate,
                          image_width=image_width, image_height=image_height)

        # train model
        history = model.fit(
            training_set,
            validation_data=validation_set,
            epochs=self.epochs,
            callbacks=[tensorboard_callback]
        )

        if plot:
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs_range = range(self.epochs)

            plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, acc, label='Training Accuracy')
            plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.title('Training and Validation Accuracy')

            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.show()

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
