from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.common import IMAGE_SIZE, load_class_names


st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to classify its plant disease.")

model_path = Path("artifacts/best_model.keras")
classes_path = Path("artifacts/class_names.json")

if not model_path.exists() or not classes_path.exists():
    st.warning("Train the model first with: python src/train.py")
    st.stop()

model = tf.keras.models.load_model(model_path)
classes = load_class_names(classes_path)
uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    x = np.expand_dims(np.asarray(image.resize(IMAGE_SIZE), dtype=np.float32), axis=0)
    probabilities = model.predict(x, verbose=0)[0]
    index = int(np.argmax(probabilities))
    st.success(f"Prediction: {classes[index]}")
    st.metric("Confidence", f"{probabilities[index] * 100:.2f}%")
    st.bar_chart({classes[i]: float(probabilities[i]) for i in np.argsort(probabilities)[-5:][::-1]})
