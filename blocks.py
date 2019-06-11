import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, add


def plain(inputs: tf.Tensor, filters: int, conv_layers=2, cs=True, reduce=False) -> tf.Tensor:
    stride = 2 if reduce else 1
    x = Conv2D(filters, 3, stride, "same")(inputs)
    x = BatchNormalization(center=cs, scale=cs)(x)
    x = Activation("relu")(x)

    for _ in range(conv_layers-1):
        x = Conv2D(filters, 3, 1, "same")(x)
        x = BatchNormalization(center=cs, scale=cs)(x)
        x = Activation("relu")(x)

    return x


def plain_prebn(inputs: tf.Tensor, filters: int, conv_layers=2, cs=True, reduce=False) -> tf.Tensor:
    stride = 2 if reduce else 1
    x = BatchNormalization(center=cs, scale=cs)(inputs)
    x = Conv2D(filters, 3, stride, "same", activation="relu")(x)

    for _ in range(conv_layers-1):
        x = BatchNormalization(center=cs, scale=cs)(x)
        x = Conv2D(filters, 3, 1, "same", activation="relu")(x)

    return x


def resblock(inputs: tf.Tensor, filters: int, conv_layers=2, cs=True, reduce=False) -> tf.Tensor:
    stride = 2 if reduce else 1

    x = Conv2D(filters, 3, stride, "same")(inputs)
    x = BatchNormalization(center=cs, scale=cs)(x)
    x = Activation("relu")(x)

    for _ in range(conv_layers-2):
        x = Conv2D(filters, 3, 1, "same")(x)
        x = BatchNormalization(center=cs, scale=cs)(x)
        x = Activation("relu")(x)

    x = Conv2D(filters, 3, 1, "same")(x)
    x = BatchNormalization(center=cs, scale=cs)(x)

    if reduce:
        inputs = Conv2D(filters, 3, stride, "same")(inputs)

    x = add([inputs, x])

    x = Activation("relu")(x)

    return x


def resblock_prebn(inputs: tf.Tensor, filters: int, conv_layers=2, cs=True, reduce=False) -> tf.Tensor:
    stride = 2 if reduce else 1
    x = BatchNormalization(center=cs, scale=cs)(inputs)
    x = Conv2D(filters, 3, stride, "same", activation="relu")(x)

    for _ in range(conv_layers-2):
        x = BatchNormalization(center=cs, scale=cs)(x)
        x = Conv2D(filters, 3, 1, "same", activation="relu")(x)

    x = BatchNormalization(center=cs, scale=cs)(x)
    x = Conv2D(filters, 3, 1, "same")(x)

    if reduce:
        inputs = Conv2D(filters, 3, stride, "same")(inputs)

    x = add([inputs, x])

    x = Activation("relu")(x)

    return x


__all__ = ["plain",  "plain_prebn", "resblock", "resblock_prebn"]
