import matplotlib.pyplot as plt
import numpy as np
import collections

import tensorflow.keras as keras

DataFrame = collections.namedtuple('DataFrame', ['images', 'labels'])


# TODO: Not sure how to do this with Keras api directly.
# TODO: The model.fit method wants both input and output to be tensors or ndarrays.
# TODO: This helper function creates the ndarray based one-hot encoding.
def one_hot_encoding(array, n_classes):
    return np.eye(n_classes)[array]


def load_data(one_hot=False, one_hot_depth=0):
    fashion_mnist = keras.datasets.fashion_mnist

    # X_images has shape (# examples, width, height) - each containing a grayscale image (pixel values - 0 to 255)
    # X_labels has shape (# examples) - each containing a class number
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    if one_hot:
        train_labels = one_hot_encoding(train_labels, one_hot_depth)
        test_labels = one_hot_encoding(test_labels, one_hot_depth)

    return {'train': DataFrame(pre_process(train_images), train_labels),
            'test': DataFrame(pre_process(test_images), test_labels)}


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
