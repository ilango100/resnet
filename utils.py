import math
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


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


def plot_metric(histories: dict, metric: str):
    for variant, history in histories.items():
        plt.plot(history[metric], label=variant)
    plt.legend()
    plt.title(metric.capitalize())


def train_and_evaluate(model, bsize=1024, epochs=500):
    # Download and prepare dataset
    cifar = tfds.builder("cifar10")
    cifar.download_and_prepare()

    trsteps = int(
        math.ceil(cifar.info.splits[tfds.Split.TRAIN].num_examples/bsize))
    valsteps = int(
        math.ceil(cifar.info.splits[tfds.Split.TEST].num_examples/bsize))

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
        tf.keras.callbacks.EarlyStopping("val_acc", patience=50)
    ],
        validation_data=val, validation_steps=valsteps)

    # Summarize
    plot_history(hist.history)
    return hist.history


__all__ = ["plot_history", "plot_metric", "train_and_evaluate"]
