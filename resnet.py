import tensorflow as tf
import tensorflow_datasets as tfds


def resnet(block, input_shape=(32, 32, 3), filters=[64, 128, 256], blocks=2, classes=10, momentum=0.9) -> tf.keras.Model:
    inp = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(
        filters[0], 5, 2, "same", activation="relu")(inp)

    for _ in range(blocks):
        x = block(x, filters[0], momentum=momentum)

    for filt in filters[1:]:
        x = block(x, filt, reduce=True, momentum=momentum)
        for _ in range(blocks):
            x = block(x, filt, momentum=momentum)

    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(classes, activation="softmax")(x)

    return tf.keras.Model(inputs=inp, outputs=x)


def train_and_evaluate(block):
    pass


__all__ = ["resnet", "train_and_evaluate"]
