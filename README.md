# Plant Disease Detection System

An end-to-end full-stack AI application that detects plant diseases from leaf images using transfer learning with EfficientNetB0.

The application supports single-image and batch prediction, confidence scores, top-3 predictions, Grad-CAM explanations, prediction history, CSV export, and history deletion.

## Application flow

```text
User uploads leaf images
        ↓
React + Vite frontend
        ↓
FastAPI REST API
        ↓
EfficientNetB0 model
        ↓
Prediction and confidence score
        ↓
Grad-CAM explanation
        ↓
SQLite prediction history
```

## Features

- Plant disease classification using EfficientNetB0
- Single-image prediction API
- Batch prediction for up to 20 images
- Top-3 predictions with confidence scores
- Grad-CAM heatmaps for model explainability
- React frontend with drag-and-drop upload
- Responsive result cards
- SQLite prediction history
- Export prediction history as CSV
- Clear prediction history
- FastAPI Swagger documentation
- Original Streamlit prototype included

## Technology stack

| Technology | Purpose |
|---|---|
| Python | Machine-learning and backend programming |
| TensorFlow/Keras | Model training and inference |
| EfficientNetB0 | Leaf disease classification model |
| PlantVillage | Public training dataset |
| NumPy | Numerical image processing |
| Pillow | Image reading and resizing |
| OpenCV | Grad-CAM heatmap generation |
| Scikit-learn | Evaluation metrics and confusion matrix |
| FastAPI | Backend REST API |
| Uvicorn | FastAPI development server |
| React | Frontend user interface |
| Vite | React development and production build tool |
| JavaScript | Frontend state and API communication |
| CSS | Frontend styling and responsive layout |
| SQLite | Prediction-history storage |

## Dataset

The project uses the public PlantVillage dataset:

- GitHub: https://github.com/spMohanty/PlantVillage-Dataset
- TensorFlow Datasets: https://www.tensorflow.org/datasets/catalog/plant_village
- Kaggle mirror: https://www.kaggle.com/datasets/emmarex/plantdisease

The current trained model uses 15 classes from the dataset. The dataset should be arranged as one folder per class:

```text
data/plantvillage/
├── Pepper__bell___Bacterial_spot/
├── Pepper__bell___healthy/
├── Potato___Early_blight/
├── Potato___healthy/
├── Potato___Late_blight/
├── Tomato_Bacterial_spot/
├── Tomato_Early_blight/
├── Tomato_healthy/
└── ...
```

If using the GitHub repository, copy the class folders from `raw/color` into `data/plantvillage`.

## Project structure

```text
resume_project/
├── artifacts/
│   ├── best_model.keras
│   ├── final_model.keras
│   ├── class_names.json
│   └── gradcam.jpg
├── backend/
│   ├── main.py
│   ├── database.py
│   └── models/
├── data/
│   └── plantvillage/
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── common.py
├── analysis_history.db
├── app.py
├── requirements.txt
└── README.md
```

## Installation on Windows

Open PowerShell in `E:\resume_project`:

```powershell
cd E:\resume_project
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install frontend dependencies:

```powershell
cd frontend
npm install
```

## Train the model

From the project root:

```powershell
python src\train.py --data-dir data\plantvillage --epochs 10 --fine-tune-epochs 5
```

The trained files are saved in `artifacts/`:

```text
best_model.keras
final_model.keras
class_names.json
```

## Evaluate the model

```powershell
python src\evaluate.py --data-dir data\plantvillage
```

The evaluation script reports precision, recall, F1-score, and the confusion matrix.

## Test prediction from the command line

```powershell
python src\predict.py --image "path\to\leaf.jpg"
```

Generate a Grad-CAM image:

```powershell
python src\gradcam.py --image "path\to\leaf.jpg"
```

The heatmap is saved to `artifacts/gradcam.jpg`.

## Run the full-stack application

Run the backend and frontend in separate terminals.

### Terminal 1: FastAPI backend

```powershell
cd E:\resume_project
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --reload
```

Backend URLs:

- API: http://127.0.0.1:8000
- Swagger documentation: http://127.0.0.1:8000/docs

### Terminal 2: React frontend

```powershell
cd E:\resume_project\frontend
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

The backend must be running before using the React application.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Checks API and model status |
| POST | `/predict` | Predicts one image |
| POST | `/predict-batch` | Predicts up to 20 images and returns Grad-CAM heatmaps |
| GET | `/history` | Returns saved prediction history |
| DELETE | `/history` | Deletes all saved history |
| GET | `/history/export` | Downloads history as CSV |

Interactive API testing is available at:

```text
http://127.0.0.1:8000/docs
```

## Prediction history

Prediction records are stored in:

```text
analysis_history.db
```

Each record contains:

- Database ID
- Uploaded filename
- Predicted class
- Confidence score
- UTC timestamp

The React frontend provides History, Export CSV, and Clear History controls.

## Optional GPU setup

Modern TensorFlow versions do not use NVIDIA CUDA directly from native Windows. GPU training can be configured through WSL2.

Inside Ubuntu/WSL2:

```bash
cd /mnt/e/resume_project
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
pip install --upgrade pip
pip install "tensorflow[and-cuda]>=2.18,<2.19"
```

Check GPU detection:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

For a 4 GB GPU, use a smaller batch size such as `BATCH_SIZE = 16` in `src/common.py`.

## Streamlit prototype

`app.py` is the original Streamlit prototype. The main application now uses React and FastAPI.

To run the prototype:

```powershell
streamlit run app.py
```

## Troubleshooting

### Frontend cannot connect to backend

Make sure the FastAPI server is running:

```powershell
python -m uvicorn backend.main:app --reload
```

### `npm run dev` cannot find `package.json`

Run the command from the frontend directory:

```powershell
cd E:\resume_project\frontend
npm run dev
```

### Model not found

Confirm these files exist:

```text
artifacts/best_model.keras
artifacts/class_names.json
```

### Dataset not detected

The class folders must be directly inside `data/plantvillage`, not inside another nested folder.

## Disclaimer

This application is an educational research prototype. Its predictions should not replace professional agricultural diagnosis or treatment advice.
