import matplotlib.pyplot as plt
from matplotlib.image import imread

import numpy as np
import collections

import tensorflow as tf
import tensorflow.keras as keras

DataFrame = collections.namedtuple('DataFrame', ['images', 'labels'])


def load_data(manifest, dims):
    filenames, _, labels = parse_manifest(manifest)
    images = np.asarray(list(map(lambda f: np.reshape(imread(f), dims), filenames)))
    labels = np.asarray([[float(c) for c in list(s)] for s in labels])

    return images, labels


def parse_manifest(path):
    with open(path) as fd:
        images_filenames = []
        mask_filenames = []
        categories = []

        for line in fd:
            tokens = line.strip().split(" ")

            images_filenames.append(tokens[0])
            mask_filenames.append(tokens[1])
            categories.append(tokens[2])

    return images_filenames, mask_filenames, categories


# def dataset_inputs(image_filenames, label_filenames, batch_size, INPUT_DIMS):
#     images = ops.convert_to_tensor(image_filenames, dtype=dtypes.string)
#     labels = ops.convert_to_tensor(label_filenames, dtype=dtypes.string)
#
#     filename_queue = tf.train.slice_input_producer([images, labels], shuffle=True)
#
#     image, label = dataset_reader(filename_queue, width, height, channels)
#     reshaped_image = tf.cast(image, tf.float32)
#     min_queue_examples = 300
#     print('Filling queue with %d input images before starting to train. '
#           'This may take some time.' % min_queue_examples)
#
#     # Generate a batch of images and labels by building up a queue of examples.
#     return _generate_image_and_label_batch(reshaped_image, label,
#                                            min_queue_examples, batch_size,
#                                            shuffle=True)


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=1,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=1,
            capacity=min_queue_examples + 3 * batch_size)

    tf.summary.image('training_images', images)
    return images, label_batch


def show_image(image, interactive=False):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)

    # Only needed for non-interactive sessions
    if not interactive:
        plt.show()


def pre_process(images):
    # Scaling pixel values before feeding into the neural network
    return images / 255.0


def show_image_matrix(images, labels, names, interactive=False):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.get_cmap('binary'))
        plt.xlabel(names[labels[i]])

    # Only needed for non-interactive sessions
    if not interactive:
        plt.show()
