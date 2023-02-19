from datasets.auto_mpg_dataset import get_dataset
from models.regression import get_model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()


class TrainerRegression:
    def __init__(self, epochs, learning_rate, val_split, train_split, **kwargs):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.val_split = val_split
        self.train_split = train_split

    def train(self, plot=True, **kwargs):
        # get dataset
        x_train, y_train, x_test, y_test = get_dataset(split=self.train_split)

        # get model
        model = get_model(train_features=x_train, lr=self.learning_rate)

        # tensorboard callback to log model training
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./log")

        # train model
        loss_history = model.fit(
            x_train, y_train,
            validation_split=self.val_split,
            epochs=self.epochs,
            callbacks=[tensorboard_callback]
        )

        # show one of the input vectors
        x_1 = pd.DataFrame(x_test).iloc[0]
        print(x_1)

        # Make a prediction (miles per gallon)
        #        cyl disp horsepower weight accel modelyear europe japan usa
        x_pred = np.array([6.0, 250.0, 400.0, 1500.0, 9.5, 95.0, 1.0, 0.0, 0.0])
        print("Predict on base of input:")
        #y_pred = model.predict(x_pred, batch_size=1).flatten()
        y_pred = model(x_pred, training=False).numpy()[0][0]
        print(y_pred)

        y_predict = model.predict(x_test).flatten()
        error = y_predict - y_test

        if plot:
            model.summary()
            plot_loss(loss_history)

            plt.hist(error, bins=25)
            plt.xlabel('Prediction Error [MPG]')
            _ = plt.ylabel('Count')
            plt.show()
