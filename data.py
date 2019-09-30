import math
from os import sep
from os.path import join
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds


def split_trainval(dataset, l, frac=0.8):
    trsteps = math.ceil(frac*l)

    val = dataset.skip(trsteps)
    train = dataset.take(trsteps)
    return train, val, trsteps


def cifar10(datapath, bsize):
    # Download and prepare dataset
    cifar = tfds.builder("cifar10", data_dir=datapath)
    cifar.download_and_prepare()

    # Get epoch steps
    totsteps = cifar.info.splits[tfds.Split.TRAIN].num_examples
    trsteps = math.ceil(totsteps * 0.8)
    valsteps = totsteps - trsteps
    testeps = cifar.info.splits[tfds.Split.TEST].num_examples

    # Get data
    train, val = tfds.Split.TRAIN.subsplit(weighted=[4, 1])
    train = cifar.as_dataset(split=train, as_supervised=True)
    val = cifar.as_dataset(split=val, as_supervised=True)
    test = cifar.as_dataset(split=tfds.Split.TEST, as_supervised=True)

    def augment(x, y):
        x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
        x = tf.image.random_crop(x, [32, 32, 3])
        x = tf.image.random_flip_left_right(x)
        return x, y

    # Apply operations
    train = train.map(augment).shuffle(trsteps, reshuffle_each_iteration=True).batch(
        bsize).repeat().prefetch(tf.data.experimental.AUTOTUNE)
    val = val.batch(bsize).repeat().prefetch(tf.data.experimental.AUTOTUNE)
    test = test.batch(bsize).repeat().prefetch(tf.data.experimental.AUTOTUNE)

    # Epoch steps
    trsteps = math.ceil(trsteps/bsize)
    valsteps = math.ceil(valsteps/bsize)
    testeps = math.ceil(testeps/bsize)

    return train, val, test, trsteps, valsteps, testeps, 10


def tiny_imagenet(datapath, bsize):
    # Download and prepare dataset
    cifar = tfds.builder("tiny_imagenet", data_dir=datapath)
    cifar.download_and_prepare()

    # Get epoch steps
    totsteps = cifar.info.splits[tfds.Split.TRAIN].num_examples
    trsteps = math.ceil(totsteps * 0.8)
    valsteps = totsteps - trsteps
    testeps = cifar.info.splits[tfds.Split.TEST].num_examples

    # Get data
    train = cifar.as_dataset(split="train[:80%]", as_supervised=True)
    val = cifar.as_dataset(split="train[80%:]", as_supervised=True)
    test = cifar.as_dataset(split="validation", as_supervised=True)

    # Apply operations
    train = train.repeat().batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)
    val = val.repeat().batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)
    test = test.repeat().batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)

    # Epoch steps
    trsteps = math.ceil(trsteps/bsize)
    valsteps = math.ceil(valsteps/bsize)
    testeps = math.ceil(testeps/bsize)

    return train, val, test, trsteps, valsteps, testeps, 200


__all__ = ["cifar10", "tiny_imagenet"]
