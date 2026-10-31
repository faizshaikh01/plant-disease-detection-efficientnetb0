import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from common import BATCH_SIZE, IMAGE_SIZE, SEED, load_class_names


def main(args):
    model = tf.keras.models.load_model(args.model)
    classes = load_class_names(args.classes)
    dataset = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )
    y_true = np.concatenate([labels.numpy() for _, labels in dataset])
    y_prob = model.predict(dataset, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    label_ids = list(range(len(classes)))
    print(
        classification_report(
            y_true,
            y_pred,
            labels=label_ids,
            target_names=classes,
            zero_division=0,
        )
    )
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=label_ids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/plantvillage")
    parser.add_argument("--model", default="artifacts/best_model.keras")
    parser.add_argument("--classes", default="artifacts/class_names.json")
    main(parser.parse_args())
