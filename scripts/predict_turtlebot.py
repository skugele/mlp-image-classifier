import tensorflow as tf
import numpy as np

from mlp_turtlebot import create_model, CHECKPOINT_DIR, cleanup, load_data, INPUT_DIMS, BATCH_SIZE, TEST_MANIFEST, \
    CLASSES, VALIDATE_MANIFEST


def main(argv=None):
    images, labels = load_data(VALIDATE_MANIFEST, INPUT_DIMS)
    data_size = len(images)

    model = create_model()

    # Restore previously trained model weights (if they exist)
    chkpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if chkpt:
        model.load_weights(chkpt)
    else:
        raise RuntimeError('Predictions require a trained model!')

    predictions = (model.predict(images) > 0.8).astype(int)
    # matches = predictions - labels.astype(int)
    for c in range(len(CLASSES)):
        n_correct = np.sum((predictions[:, c] == labels[:, c]).astype(int))
        print('Class \"{}\" accuracy: {}'.format(CLASSES[c], np.float(n_correct) / np.float(data_size)))


if __name__ == '__main__':
    tf.app.run()
