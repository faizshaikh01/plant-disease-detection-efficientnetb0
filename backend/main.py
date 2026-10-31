from pathlib import Path
import json
import io
import base64
import csv
from io import StringIO

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

from .database import clear_history, get_history, init_db, save_prediction


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "artifacts" / "best_model.keras"
CLASSES_PATH = ROOT_DIR / "artifacts" / "class_names.json"
IMAGE_SIZE = (224, 224)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not CLASSES_PATH.exists():
    raise FileNotFoundError(f"Class names not found: {CLASSES_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
class_names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
init_db()

app = FastAPI(
    title="Plant Disease Detection API",
    description="EfficientNetB0-based plant disease prediction service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH.name,
        "classes": len(class_names),
    }


@app.get("/history")
def history(limit: int = 50):
    safe_limit = max(1, min(limit, 100))
    return {"items": get_history(safe_limit)}


@app.delete("/history")
def delete_history():
    clear_history()
    return {"status": "ok", "message": "Prediction history cleared"}


@app.get("/history/export")
def export_history():
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "filename", "prediction", "confidence", "created_at"])
    for item in get_history(10000):
        writer.writerow([item["id"], item["filename"], item["class_name"], item["confidence"], item["created_at"]])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=prediction_history.csv"},
    )


def predict_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    resized = image.resize(IMAGE_SIZE)
    input_array = np.asarray(resized, dtype=np.float32)
    input_array = np.expand_dims(input_array, axis=0)
    probabilities = model.predict(input_array, verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]
    predictions = [
        {
            "class_name": class_names[int(index)],
            "confidence": round(float(probabilities[index]) * 100, 2),
        }
        for index in top_indices
    ]
    return predictions, int(top_indices[0]), input_array


def gradcam_image(input_array, class_index):
    base = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    conv_layer = next(layer for layer in reversed(base.layers) if isinstance(layer, tf.keras.layers.Conv2D))
    feature_model = tf.keras.models.Model(base.input, [conv_layer.output, base.output])
    with tf.GradientTape() as tape:
        x = input_array
        conv_output = None
        for layer in model.layers[1:]:
            if layer.name == base.name:
                conv_output, x = feature_model(x, training=False)
            else:
                x = layer(x, training=False)
        loss = x[:, class_index]
    gradients = tape.gradient(loss, conv_output)
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(conv_output * weights[:, None, None, :], axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    heatmap = cv2.resize(heatmap.numpy(), IMAGE_SIZE)
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    original = np.uint8(input_array[0])
    overlay = cv2.addWeighted(cv2.cvtColor(original, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
    success, encoded = cv2.imencode('.jpg', overlay)
    if not success:
        return None
    return base64.b64encode(encoded.tobytes()).decode('utf-8')


async def read_image(file: UploadFile):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file")
    try:
        image_bytes = await file.read()
        Image.open(io.BytesIO(image_bytes)).verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image")
    return image_bytes


def result_for_file(file: UploadFile, predictions):
    return {
        "filename": file.filename,
        "prediction": predictions[0],
        "top_predictions": predictions,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await read_image(file)
    predictions, _, _ = predict_image(image_bytes)
    result = result_for_file(file, predictions)
    result["analysis"] = save_prediction(file.filename, predictions[0])
    return result


@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="You can upload up to 20 images at a time")
    results = []
    for file in files:
        image_bytes = await read_image(file)
        predictions, class_index, input_array = predict_image(image_bytes)
        result = result_for_file(file, predictions)
        result['gradcam'] = gradcam_image(input_array, class_index)
        result['analysis'] = save_prediction(file.filename, predictions[0])
        results.append(result)
    return {"count": len(results), "results": results}
