import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from common import IMAGE_SIZE, load_class_names


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return None, layer.name
        if isinstance(layer, tf.keras.Model):
            for nested in reversed(layer.layers):
                if isinstance(nested, tf.keras.layers.Conv2D):
                    return layer.name, nested.name
    raise ValueError("No convolutional layer found")


def make_gradcam(model, image_tensor, class_index, base_name, layer_name):
    base = model.get_layer(base_name) if base_name else None
    feature_model = None
    if base is not None:
        conv_layer = base.get_layer(layer_name)
        feature_model = tf.keras.models.Model(
            base.input, [conv_layer.output, base.output]
        )
    with tf.GradientTape() as tape:
        x = image_tensor
        conv_output = None
        for layer in model.layers[1:]:
            if base is not None and layer.name == base.name:
                conv_output, x = feature_model(x, training=False)
            else:
                x = layer(x, training=False)
        predictions = x
        loss = predictions[:, class_index]
    gradients = tape.gradient(loss, conv_output)
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    cam = tf.reduce_sum(tf.multiply(conv_output, weights[:, None, None, :]), axis=-1)
    cam = tf.maximum(cam, 0)[0]
    cam /= tf.reduce_max(cam) + tf.keras.backend.epsilon()
    return cam.numpy()


def main(args):
    model = tf.keras.models.load_model(args.model)
    classes = load_class_names(args.classes)
    original = Image.open(args.image).convert("RGB")
    resized = original.resize(IMAGE_SIZE)
    tensor = np.expand_dims(np.asarray(resized, dtype=np.float32), 0)
    probabilities = model.predict(tensor, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    base_name, layer_name = find_last_conv_layer(model)
    if args.layer:
        layer_name = args.layer
    heatmap = make_gradcam(model, tensor, class_index, base_name, layer_name)
    heatmap = cv2.resize(heatmap, original.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    base = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(base, 0.6, heatmap, 0.4, 0)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, overlay)
    print(f"Prediction: {classes[class_index]} ({probabilities[class_index] * 100:.2f}%)")
    print(f"Grad-CAM saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/best_model.keras")
    parser.add_argument("--classes", default="artifacts/class_names.json")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="artifacts/gradcam.jpg")
    parser.add_argument("--layer", default=None)
    main(parser.parse_args())
