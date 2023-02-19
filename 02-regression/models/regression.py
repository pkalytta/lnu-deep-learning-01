import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


class Regression(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(self, train_features):
        super(Regression, self).__init__()
        self.normalizer = preprocessing.Normalization()
        self.normalizer.adapt(np.array(train_features))

        self.dense1 = tf.keras.layers.Dense(128, activation='relu', name='dense1')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu', name='dense2')
        self.out = tf.keras.layers.Dense(1, activation='linear', name='out')

    def call(self, inputs, **kwargs):
        x = self.normalizer(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.out(x)


def get_model(train_features, lr=0.001):
    model = Regression(train_features)
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(lr))
    return model
