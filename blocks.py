import tensorflow as tf


def plain(inputs: tf.Tensor, filters: int, reduce=False, momentum=0.9) -> tf.Tensor:
    stride = 2 if reduce else 1
    x = tf.keras.layers.Conv2D(filters, 3, stride, "same")(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(filters, 3, 1, "same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.Activation("relu")(x)

    return x


def plain_prebn(inputs: tf.Tensor, filters: int, reduce=False, momentum=0.9) -> tf.Tensor:
    stride = 2 if reduce else 1
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(inputs)
    x = tf.keras.layers.Conv2D(
        filters, 3, stride, "same", activation="relu")(x)

    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.Conv2D(filters, 3, 1, "same", activation="relu")(x)

    return x
