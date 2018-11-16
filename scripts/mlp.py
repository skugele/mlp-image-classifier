# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import collections
from util import show_image, pre_process, show_image_matrix

CHECKPOINT_DIR = 'checkpoints'
LOGS_DIR = 'logs'

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

DataFrame = collections.namedtuple('DataFrame', ['images', 'labels'])


def load_data():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # X_images has shape (# examples, width, height) - each containing a grayscale image (pixel values - 0 to 255)
    # X_labels has shape (# examples) - each containing a class number
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    return {'train': DataFrame(pre_process(train_images), train_labels),
            'test': DataFrame(pre_process(test_images), test_labels)}


def create_model(summary=False):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # Use sigmoid units in the output layer and then use "binary_crossentrpy" loss.
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    if summary:
        model.summary()

    return model


def train_model(model, images, labels, epochs):
    # Callback to save model during training
    chkpt_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(CHECKPOINT_DIR, 'cp-{epoch:08d}.ckpt'),
                                                        save_weights_only=True,
                                                        verbose=1,
                                                        period=1)

    # Callback to save logs for tensorboard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(LOGS_DIR, 'tensorboard'),
                                                          histogram_freq=0, write_graph=True, write_grads=False,
                                                          write_images=False)

    model.fit(images, labels, epochs=epochs, callbacks=[chkpt_callback, tensorboard_callback])


def eval_model(model, images, labels):
    loss, accuracy = model.evaluate(images, labels)
    return loss, accuracy


def cleanup():
    # Added to suppress an exception that was occurring at the end of program execution
    # when Keras was trying to delete the backing session.  This line was recommended
    # in discussion forum here: https://github.com/keras-team/keras/issues/2102
    tf.keras.backend.clear_session()


def main(argv=None):
    if not tf.gfile.Exists(CHECKPOINT_DIR):
        tf.gfile.MakeDirs(CHECKPOINT_DIR)

    if not tf.gfile.Exists(CHECKPOINT_DIR):
        tf.gfile.MakeDirs(CHECKPOINT_DIR)

    dataset = load_data()
    model = create_model()

    # Restore previously trained model weights (if they exist)
    chkpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if chkpt:
        model.load_weights(chkpt)

    train_model(model, dataset['train'].images, dataset['train'].labels, epochs=5)

    cleanup()


if __name__ == '__main__':
    tf.app.run()
