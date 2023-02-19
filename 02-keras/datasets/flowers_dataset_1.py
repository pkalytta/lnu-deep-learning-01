import tensorflow as tf
import tensorflow_datasets as tfds


def get_dataset(split=.2):
    def preprocess(ds):
        x = ds['image']
        x = tf.cast(x, tf.float32)
        y = ds['label']
        return x, y
    ds_full, ds_full_info = tfds.load(name="tf_flowers", split="train", with_info=True)
    ds_full = ds_full.map(preprocess)
    ds_full = ds_full.batch(1)
    return ds_full


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data_set = get_dataset()
    for i, (x_, y_) in enumerate(data_set):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(x_[0].numpy().astype("uint8"))
        plt.title(y_[0].numpy())
        plt.axis("off")
        if i >= 8:
            break

    plt.show()
    print()
