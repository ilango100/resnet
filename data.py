import math
import tensorflow as tf
import tensorflow_datasets as tfds


def split_trainval(dataset, l, frac=0.8):
    trsteps = math.ceil(frac*l)

    val = dataset.skip(trsteps)
    train = dataset.take(trsteps)
    return train, val, trsteps


def cifar10(bsize):
    # Download and prepare dataset
    cifar = tfds.builder("cifar10")
    cifar.download_and_prepare()

    # Get epoch steps
    totsteps = cifar.info.splits[tfds.Split.TRAIN].num_examples
    trsteps = math.ceil(totsteps * 0.8)
    valsteps = totsteps - trsteps
    testeps = cifar.info.splits[tfds.Split.TEST].num_examples

    # Get data
    # train, val = tfds.Split.TRAIN.subsplit(weighted=[4, 1])
    train, val = "train[:80%]", "train[80%:]"
    train, val, test = [cifar.as_dataset(split=x, as_supervised=True) for x in [
        train, val, tfds.Split.TEST]]

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


def cats_vs_dogs(bsize):
    cvd = tfds.builder("cats_vs_dogs")
    cvd.download_and_prepare()

    totsteps = cvd.info.splits[tfds.Split.TRAIN].num_examples
    trsteps = math.ceil(totsteps*0.7)
    valsteps = math.ceil(totsteps*0.15)
    testeps = totsteps - trsteps - valsteps

    train, val, test = "train[:70%]", "train[70%:85%]", "train[85%:]"
    train, val, test = [cvd.as_dataset(split=x, batch_size=bsize,
                                       shuffle_files=True, as_supervised=True) for x in [train, val, test]]

    # Per epoch steps
    trsteps = math.ceil(trsteps/bsize)
    valsteps = math.ceil(valsteps/bsize)
    testeps = math.ceil(testeps/bsize)

    return train, val, test, trsteps, valsteps, testeps, 2


def imagenette(bsize):
    imt = tfds.builder("imagenette/160px")
    imt.download_and_prepare()

    totsteps = imt.info.splits[tfds.Split.TRAIN].num_examples
    trsteps = math.ceil(totsteps * 0.8)
    valsteps = totsteps - trsteps
    testeps = imt.info.splits[tfds.Split.VALIDATION].num_examples

    train, val, test = "train[:80%]", "train[80%:]", "validation"
    train, val, test = [imt.as_dataset(split=x, batch_size=bsize,
                                       shuffle_files=True, as_supervised=True) for x in [train, val, test]]

    trsteps, valsteps, testeps = [
        math.ceil(x/bsize) for x in [trsteps, valsteps, testeps]]

    return train, val, test, trsteps, valsteps, testeps, 10


def tiny_imagenet(bsize):
    # Download and prepare dataset
    cifar = tfds.builder("tiny_imagenet")
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


__all__ = ["cifar10", "cats_vs_dogs", "imagenette", "tiny_imagenet"]
