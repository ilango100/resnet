import math
import tensorflow as tf
import tensorflow_datasets as tfds


def split_trainval(dataset, l, frac=0.8):
    trsteps = math.ceil(frac*l)

    val = dataset.skip(trsteps)
    train = dataset.take(trsteps)
    return train, val, trsteps


def std(x, y):
    return x/255, y


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
    splits = ("train[:80%]", "train[80%:]", "test")
    train, val, test = [cifar.as_dataset(split=x,
                                         shuffle_files=True,
                                         batch_size=bsize,
                                         as_supervised=True)
                        for x in splits]

    def augment(x, y):
        x = tf.pad(x, [[0, 0], [4, 4], [4, 4], [0, 0]])
        x = tf.image.random_crop(x, [bsize, 32, 32, 3])
        x = tf.image.random_flip_left_right(x)
        return x, y

    # Apply operations
    train = train \
        .map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .map(std, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .prefetch(tf.data.experimental.AUTOTUNE).repeat()
    val = val \
        .map(std, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .prefetch(tf.data.experimental.AUTOTUNE).repeat()
    test = test \
        .map(std, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .prefetch(tf.data.experimental.AUTOTUNE).repeat()

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
    train, val, test = [cvd.as_dataset(split=x,
                                       shuffle_files=True,
                                       batch_size=bsize,
                                       as_supervised=True)
                        for x in [train, val, test]]

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
    train, val, test = [imt.as_dataset(split=x,
                                       shuffle_files=True,
                                       batch_size=bsize,
                                       as_supervised=True)
                        for x in [train, val, test]]

    def augment(x, y):
        x = tf.pad(x, [[0, 0], [20, 20], [20, 20], [0, 0]])
        x = tf.image.random_crop(x, (160, 160))
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, 0.2)
        return x, y

    train = train \
        .map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .map(std, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val = val.map(std, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = test.map(std, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    trsteps, valsteps, testeps = [
        math.ceil(x/bsize) for x in [trsteps, valsteps, testeps]]

    return train, val, test, trsteps, valsteps, testeps, 10


# tiny_imagenet not implemented yet
# def tiny_imagenet(bsize):
#     # Download and prepare dataset
#     cifar = tfds.builder("tiny_imagenet")
#     cifar.download_and_prepare()

#     # Get epoch steps
#     totsteps = cifar.info.splits[tfds.Split.TRAIN].num_examples
#     trsteps = math.ceil(totsteps * 0.8)
#     valsteps = totsteps - trsteps
#     testeps = cifar.info.splits[tfds.Split.TEST].num_examples

#     # Get data
#     train = cifar.as_dataset(split="train[:80%]", as_supervised=True)
#     val = cifar.as_dataset(split="train[80%:]", as_supervised=True)
#     test = cifar.as_dataset(split="validation", as_supervised=True)

#     # Apply operations
#     train = train.repeat().batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)
#     val = val.repeat().batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)
#     test = test.repeat().batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)

#     # Epoch steps
#     trsteps = math.ceil(trsteps/bsize)
#     valsteps = math.ceil(valsteps/bsize)
#     testeps = math.ceil(testeps/bsize)

#     return train, val, test, trsteps, valsteps, testeps, 200


__all__ = ["cifar10", "cats_vs_dogs", "imagenette"]
