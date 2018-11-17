import os

import tensorflow as tf
import tensorflow.keras as keras

from util_turtlebot import load_data

INPUT_DIMS = (160, 120, 3)  # Width x Height
N_EPOCHS = 1000
BATCH_SIZE = 16

CHECKPOINT_DIR = 'checkpoints'
LOGS_DIR = 'logs'

TRAIN_MANIFEST = '/var/local/data/skugele/COMP8150/project/images/train.txt'
TEST_MANIFEST = '/var/local/data/skugele/COMP8150/project/images/test.txt'
VALIDATE_MANIFEST = '/var/local/data/skugele/COMP8150/project/images/validate.txt'

CLASSES = ['Background', 'Cylinder', 'Cube', 'Sphere']


def create_model(summary=True):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=INPUT_DIMS),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dense(len(CLASSES), activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    if summary:
        model.summary()

    return model


def train_model(model, dataset, epochs, steps_per_epoch):
    # Callback to save model during training
    chkpt_callback = keras.callbacks.ModelCheckpoint(os.path.join(CHECKPOINT_DIR, 'cp-{epoch:08d}.ckpt'),
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     period=1)

    # Callback to save logs for tensorboard
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=os.path.join(LOGS_DIR, 'tensorboard'),
                                                       histogram_freq=0, write_graph=True, write_grads=False,
                                                       write_images=False)

    model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[chkpt_callback, tensorboard_callback])


def eval_model(model, images, labels):
    loss, accuracy = model.evaluate(images, labels)
    return loss, accuracy


def cleanup():
    # Added to suppress an exception that was occurring at the end of program execution
    # when Keras was trying to delete the backing session.  This line was recommended
    # in discussion forum here: https://github.com/keras-team/keras/issues/2102
    keras.backend.clear_session()


def main(argv=None):
    if not tf.gfile.Exists(CHECKPOINT_DIR):
        tf.gfile.MakeDirs(CHECKPOINT_DIR)

    if not tf.gfile.Exists(LOGS_DIR):
        tf.gfile.MakeDirs(LOGS_DIR)

    images, labels = load_data(TRAIN_MANIFEST, INPUT_DIMS)
    data_size = len(images)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(BATCH_SIZE).repeat()

    model = create_model()

    # Restore previously trained model weights (if they exist)
    chkpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if chkpt:
        model.load_weights(chkpt)

    train_model(model, dataset, epochs=N_EPOCHS, steps_per_epoch=int(data_size / BATCH_SIZE))

    cleanup()


if __name__ == '__main__':
    tf.app.run()
