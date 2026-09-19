# 🐾 Dog Breed Classifier – Full-Stack Web App

AI-powered dog breed identification using a MobileNetV2-based TensorFlow model trained on the **Stanford Dogs Dataset** (120 breeds).

---

## 📁 Project Structure

```
dog-breed-webapp/
├── main.py              # FastAPI app – /predict and /health endpoints
├── model_utils.py       # Model loading, image preprocessing, inference
├── schemas.py           # Pydantic v2 request/response models
├── breeds.py            # 120 Stanford Dogs class labels + decode helper
├── requirements.txt     # Python dependencies
├── static/
│   ├── index.html       # Premium dark-themed UI
│   ├── style.css        # Glassmorphism styling + animations
│   └── script.js        # Fetch API + drag-drop + results rendering
└── model/               # ← Place your model file here
    └── dog_breed_model.h5   (or .keras / .joblib)
```

---

## ⚡ Quick Start

### 1. Clone / download this folder

```bash
cd dog-breed-webapp
```

### 2. Place your model file

Copy your trained model into the `model/` directory:

```
model/dog_breed_model.h5        # Keras HDF5
# or
model/dog_breed_model.keras     # Native Keras v3
# or
model/dog_vision_full_image_set_mobilenet_v2_Adam.joblib  # joblib-serialised
```

> **Tip**: The model file path is configurable via the `MODEL_PATH` environment variable:
> ```bash
> set MODEL_PATH=./model/my_custom_model.h5   # Windows
> export MODEL_PATH=./model/my_custom_model.h5  # macOS/Linux
> ```

### 3. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> TensorFlow is a large package (~500 MB). Installation may take a few minutes.

### 5. Run the development server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open your browser at **http://localhost:8000**

---

## 🔌 API Reference

### `GET /health`
Returns model status.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "./model/dog_breed_model.h5"
}
```

### `POST /predict`
Accepts `multipart/form-data` with a single `file` field (any common image format).

**Success (200)**:
```json
{
  "success": true,
  "top_breed": "Golden Retriever",
  "top_confidence": 0.9142,
  "predictions": [
    { "rank": 1, "breed": "Golden Retriever", "confidence": 0.9142, "confidence_pct": 91.42 },
    { "rank": 2, "breed": "Labrador Retriever", "confidence": 0.0621, "confidence_pct": 6.21 },
    { "rank": 3, "breed": "Flat-coated Retriever", "confidence": 0.0201, "confidence_pct": 2.01 }
  ]
}
```

**Error (422)** – invalid image:
```json
{
  "success": false,
  "detail": "Unsupported file type 'application/pdf'.",
  "errors": [{ "loc": ["body", "file"], "msg": "Invalid content type", "type": "value_error.image_type" }]
}
```

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Architecture | MobileNetV2 (transfer learning) |
| Dataset | Stanford Dogs – 120 breeds |
| Input size | 224 × 224 × 3 |
| Output | 120-class softmax probabilities |
| Serialisation | `.h5` / `.keras` / `.joblib` |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `Model not loaded` on health check | Ensure model file exists at `MODEL_PATH` and restart server |
| `ModuleNotFoundError: tensorflow` | Run `pip install tensorflow` |
| `ModuleNotFoundError: dill` | Run `pip install dill` (required for `.joblib` models) |
| Port 8000 in use | Use `--port 8001` in the uvicorn command |
| Slow first prediction | Normal – TF JIT-compiles the graph on the first call |

---

## 🛠 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `./model/dog_breed_model.h5` | Path to the trained Keras model |

---

## 📦 Dependencies

```
fastapi          – Web framework
uvicorn          – ASGI server
python-multipart – Multipart form parsing
pydantic         – Request/response validation
tensorflow       – Model inference
pillow           – Image decoding and resizing
numpy            – Tensor operations
joblib           – Joblib model loading (fallback)
python-dotenv    – .env file support
```
