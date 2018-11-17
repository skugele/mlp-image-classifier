from mlp import create_model, CHECKPOINT_DIR, cleanup, load_data

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

def main(argv=None):
    dataset = load_data()
    model = create_model()

    # Restore previously trained model weights (if they exist)
    chkpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if chkpt:
        model.load_weights(chkpt)
    else:
        raise RuntimeError('Predictions require a trained model!')

    test_images = dataset['test'].images[0:10]
    test_labels = dataset['test'].labels[0:10]

    predictions = (model.predict(test_images) > 0.8).astype(int)

    print(predictions)
    print(test_labels.astype(int))

    cleanup()


if __name__ == '__main__':
    tf.app.run()
