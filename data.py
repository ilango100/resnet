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

    # Apply operations
    train = train.repeat().batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)
    val = val.repeat().batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)
    test = test.repeat().batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)

    # Epoch steps
    trsteps = math.ceil(trsteps/bsize)
    valsteps = math.ceil(valsteps/bsize)
    testeps = math.ceil(testeps/bsize)

    return train, val, test, trsteps, valsteps, testeps, 10


def tiny_imagenet(datapath, bsize):
    trainpath = join(datapath, "train")
    assert tf.io.gfile.isdir(trainpath), "Train directory does not exist"
    valpath = join(datapath, "val")
    assert tf.io.gfile.isdir(valpath), "Val directory does not exist"
    classes = tf.io.gfile.listdir(trainpath)

    # Train data
    images = tf.io.gfile.glob(join(trainpath, "*", "images", "*.JPEG"))
    labels = map(lambda x: x.split(sep)[-3], images)
    labels = list(map(classes.index, labels))
    train = tf.data.Dataset.from_tensor_slices(
        (images, labels)).shuffle(len(images))
    trsteps = len(images)

    # Split train val
    train, val, trsteps = split_trainval(train, trsteps)
    valsteps = len(images)-trsteps

    # Load test data
    df = pd.read_csv(join(valpath, "val_annotations.txt"),
                     sep="\t",
                     names=["images", "label", "l", "t", "r", "b"])
    testimages = df.images.map(lambda x: join(valpath, "images", x)).tolist()
    testlabels = df.label.map(classes.index).tolist()
    test = tf.data.Dataset.from_tensor_slices(
        (testimages, testlabels)).shuffle(len(testimages))
    testeps = len(testimages)

    def f(x, y):
        img = tf.image.decode_jpeg(tf.io.read_file(x), 3)
        return img, y

    # Process datasets
    train = train.map(f).repeat().batch(
        bsize).prefetch(tf.data.experimental.AUTOTUNE)
    val = val.map(f).repeat().batch(
        bsize).prefetch(tf.data.experimental.AUTOTUNE)
    test = test.map(f).repeat().batch(
        bsize).prefetch(tf.data.experimental.AUTOTUNE)

    # Correct epoch steps
    trsteps = math.ceil(trsteps/bsize)
    valsteps = math.ceil(valsteps/bsize)
    testeps = math.ceil(testeps/bsize)

    return train, val, test, trsteps, valsteps, testeps, len(classes)


__all__ = ["cifar10", "tiny_imagenet"]
