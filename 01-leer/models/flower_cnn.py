import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers


class FlowerCNN(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(self, num_classes, image_width, image_height):
        super(FlowerCNN, self).__init__()

        self.data_augmentation = Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(image_height, image_width, 3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ])

        self.num_classes = num_classes
        self.max_pool = layers.MaxPooling2D()

        self.conv1 = layers.Conv2D(16, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(64, 3, padding='same', activation='relu')

        self.dropout = layers.Dropout(0.2)
        self.flat = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.out = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, **kwargs):
        x = self.data_augmentation(inputs)
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.flat(x)
        x = self.dense1(x)
        return self.out(x)


def get_model(num_classes, image_width, image_height, lr=0.001):
    model = FlowerCNN(num_classes, image_width, image_height)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(lr),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    import numpy as np

    model_ = get_model(num_classes=5)
    img = np.zeros((1, 180, 180, 3))
    model_(img)
    model_.summary()
