import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

from common import IMAGE_SIZE, load_class_names


def main(args):
    model = tf.keras.models.load_model(args.model)
    names = load_class_names(args.classes)
    image = Image.open(args.image).convert("RGB").resize(IMAGE_SIZE)
    x = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
    probabilities = model.predict(x, verbose=0)[0]
    top = np.argsort(probabilities)[::-1][:5]
    for index in top:
        print(f"{names[index]}: {probabilities[index] * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/best_model.keras")
    parser.add_argument("--classes", default="artifacts/class_names.json")
    parser.add_argument("--image", required=True)
    main(parser.parse_args())
