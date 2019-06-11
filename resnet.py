import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def resnet(block, input_shape=(32, 32, 3), filters=[64, 128, 256], blocks=2, classes=10) -> tf.keras.Model:
    inp = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(
        filters[0], 5, 2, "same", activation="relu")(inp)

    for _ in range(blocks):
        x = block(x, filters[0])

    for filt in filters[1:]:
        x = block(x, filt, reduce=True)
        for _ in range(blocks):
            x = block(x, filt)

    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(classes, activation="softmax")(x)

    return tf.keras.Model(inputs=inp, outputs=x)


def plot_history(history):
    # Plot loss
    plt.plot(history["loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()
    plt.show()

    # Plot accuracy
    plt.plot(history["acc"], label="Train")
    plt.plot(history["val_acc"], label="Val")
    plt.title("Accuracy")
    plt.legend()
    plt.show()

    # Print values
    print("Min val_loss:", min(history["val_loss"]))
    print("Max val_acc:", max(history["val_acc"]))


def train_and_evaluate(model, bsize=1024, epochs=100):
    # Download and prepare dataset
    cifar = tfds.builder("cifar10")
    cifar.download_and_prepare()

    trsteps = cifar.info.splits[tfds.Split.TRAIN].num_examples//bsize
    valsteps = cifar.info.splits[tfds.Split.TEST].num_examples//bsize

    train = cifar.as_dataset(split=tfds.Split.TRAIN, as_supervised=True).repeat(
    ).batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)
    val = cifar.as_dataset(split=tfds.Split.TEST, as_supervised=True).repeat(
    ).batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)

    # Compile model with default params
    model.compile("adam", "sparse_categorical_crossentropy", metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy("acc")
    ])

    # Train the model
    hist = model.fit(train, steps_per_epoch=trsteps, epochs=epochs, callbacks=[
        tf.keras.callbacks.EarlyStopping("val_acc", patience=20)
    ],
        validation_data=val, validation_steps=valsteps)

    # Summarize
    plot_history(hist.history)


__all__ = ["resnet", "plot_history", "train_and_evaluate"]
