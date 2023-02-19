import tensorflow as tf
import tensorflow_datasets as tfds


def get_dataset(split=.2, batch_size=32, image_width=180, image_height=180):
    ds_full, ds_full_info = tfds.load(name="tf_flowers", split="train", with_info=True)

    IMG_SIZE_W = image_width
    IMG_SIZE_H = image_height
    NUM_CLASSES = ds_full_info.features['label'].num_classes

    def preprocess(ds):
        x = ds['image']
        x = tf.image.resize(x, [IMG_SIZE_W, IMG_SIZE_H])  # apply resize
        x = tf.cast(x, tf.float32)
        x = x * (1. / 255)  # range between 0 and 1
        y = ds['label']
        y = tf.one_hot(y, NUM_CLASSES)  # one-hot encoding
        return x, y

    ds_full = ds_full.map(preprocess)
    ds_full = ds_full.batch(batch_size)
    return ds_full


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data_set = get_dataset(batch_size=1)
    for i, (x_, y_) in enumerate(data_set):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(x_[0].numpy())
        plt.title(y_[0].numpy())
        plt.axis("off")
        if i >= 8:
            break

    plt.show()
    print()
