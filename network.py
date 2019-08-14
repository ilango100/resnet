from tensorflow import keras


def stack(block, filters, blocks, classes, cs=True) -> keras.Model:
    inp = keras.Input(shape=(None, None, 3))

    x = keras.layers.Conv2D(filters[0], 5, 2, "same", activation="relu")(inp)

    for _ in range(blocks):
        x = block(x, filters[0], cs=cs)

    for filt in filters[1:]:
        x = block(x, filt, reduce=True, cs=cs)
        for _ in range(blocks-1):
            x = block(x, filt, cs=cs)

    x = keras.layers.GlobalAvgPool2D()(x)
    x = keras.layers.Dense(classes, activation="softmax")(x)

    return keras.Model(inputs=inp, outputs=x)


__all__ = ["stack"]
