import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, add


def plain(inputs: tf.Tensor, filters: int, reduce=False, momentum=0.9) -> tf.Tensor:
    stride = 2 if reduce else 1
    x = Conv2D(filters, 3, stride, "same")(inputs)
    x = BatchNormalization(momentum=momentum)(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, 1, "same")(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Activation("relu")(x)

    return x


def plain_prebn(inputs: tf.Tensor, filters: int, reduce=False, momentum=0.9) -> tf.Tensor:
    stride = 2 if reduce else 1
    x = BatchNormalization(momentum=momentum)(inputs)
    x = Conv2D(
        filters, 3, stride, "same", activation="relu")(x)

    x = BatchNormalization(momentum=momentum)(x)
    x = Conv2D(filters, 3, 1, "same", activation="relu")(x)

    return x


def plain_prebn_wocs(inputs: tf.Tensor, filters: int, reduce=False, momentum=0.9) -> tf.Tensor:
    stride = 2 if reduce else 1
    x = BatchNormalization(
        momentum=momentum, center=False, scale=False)(inputs)
    x = Conv2D(
        filters, 3, stride, "same", activation="relu")(x)

    x = BatchNormalization(
        momentum=momentum, center=False, scale=False)(x)
    x = Conv2D(filters, 3, 1, "same", activation="relu")(x)

    return x


__all__ = ["plain", "plain_prebn"]
