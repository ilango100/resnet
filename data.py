import math
import tensorflow as tf
import tensorflow_datasets as tfds


def split_trainval(dataset, l, frac=0.8):
    trsteps = math.ceil(frac*l)

    val = dataset.skip(trsteps)
    train = dataset.take(trsteps)
    return train, val, trsteps


def get_cifar10(bsize=1024):
    # Download and prepare dataset
    cifar = tfds.builder("cifar10")
    cifar.download_and_prepare()

    # Get epoch steps
    totsteps = cifar.info.splits[tfds.Split.TRAIN].num_examples
    train = cifar.as_dataset(split=tfds.Split.TRAIN, as_supervised=True)
    testeps = cifar.info.splits[tfds.Split.TEST].num_examples

    # Split train val
    train, val, trsteps = split_trainval(train, totsteps)
    valsteps = totsteps-trsteps

    # Process datasets
    train = train.repeat().batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)
    val = val.repeat().batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)
    test = cifar.as_dataset(split=tfds.Split.TEST,
                            as_supervised=True).repeat().batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)
    # Correct epoch steps
    trsteps = math.ceil(trsteps/bsize)
    valsteps = math.ceil(valsteps/bsize)
    testeps = math.ceil(testeps/bsize)

    return train, val, test, trsteps, valsteps, testeps


__all__ = ["get_cifar10", "split_trainval"]
