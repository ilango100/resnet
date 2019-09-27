from tensorflow import keras


def start_stack(inputs, filters, preact=False, reg=None, cs=True):
    if preact:
        return keras.layers.Conv2D(filters, 5, 2, "same", kernel_regularizer=keras.regularizers.l2(reg))(inputs)

    else:
        x = keras.layers.Conv2D(filters, 5, 2, "same")(inputs)
        x = keras.layers.BatchNormalization()(x)
        return keras.layers.Activation("relu")(x)


def stack(x, block, filters, blocks, reg=None, cs=True) -> keras.Model:
    for filt in filters:
        x = block(x, filt, reduce=True, reg=reg, cs=cs)
        for _ in range(blocks-1):
            x = block(x, filt, reg=reg, cs=cs)

    return x


def end_stack(x, classes, preact=False):
    if preact:
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalAvgPool2D()(x)
    x = keras.layers.Dense(classes, activation="softmax")(x)
    return x


__all__ = ["start_stack", "stack", "end_stack"]
