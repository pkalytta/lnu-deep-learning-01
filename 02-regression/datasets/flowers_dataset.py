import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


def get_dataset(split=.2, batch_size=32, image_width=180, image_height=180):
    split_int = str(int((split) * 100))
    (training_set, validation_set), ds_full_info = tfds.load(name="tf_flowers",
                                                             split=[
                                                                 f"train[:{split_int}%]",
                                                                 f"train[{split_int}%:]",
                                                             ],
                                                             with_info=True)
    num_classes = ds_full_info.features['label'].num_classes
    class_names = np.array(ds_full_info.features['label'].names)
    IMG_SIZE_W = image_width
    IMG_SIZE_H = image_height
    NUM_CLASSES = num_classes

    def preprocess(ds):
        x = ds['image']
        x = tf.image.resize(x, [IMG_SIZE_W, IMG_SIZE_H])  # apply resize
        x = tf.cast(x, tf.float32)
        x = x * (1. / 255)  # range between 0 and 1
        y = ds['label']
        y = tf.one_hot(y, NUM_CLASSES)  # one-hot encoding
        return x, y

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    training_set = training_set.map(preprocess)

    training_set = training_set.cache().shuffle(1000)
    training_set = training_set.prefetch(buffer_size=AUTOTUNE)
    training_set = training_set.batch(batch_size)

    validation_set = validation_set.map(preprocess)
    validation_set = validation_set.batch(batch_size)
    validation_set = validation_set.cache().prefetch(buffer_size=AUTOTUNE)

    return training_set, validation_set, num_classes, class_names


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    training_set, validation_set, num_classes = get_dataset(batch_size=1)
    for i, (x_, y_) in enumerate(training_set):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(x_[0].numpy())
        plt.title(y_[0].numpy())
        plt.axis("off")
        if i >= 8:
            break

    plt.show()
    print()
