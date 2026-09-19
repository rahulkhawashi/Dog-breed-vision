# 📦 Export Your Model from Google Colab

## Why Re-Export?

The current `.joblib` file was saved as a **dill-serialised Python function** — it contains
a snapshot of the Colab process memory, not just the model weights. This format:

- Is platform-specific (Colab = Linux; your PC = Windows)
- Is TF-version-specific (Colab TF ≠ local TF 2.20)
- **Segfaults** when loaded on a different machine

The fix is a one-time re-export from your Colab notebook using the standard Keras format.

---

## Step-by-Step: Save Your Model Properly

### 1. Open your Colab notebook and run these cells

```python
# Cell 1 – Confirm your model variable name
# (look for the variable where you called model.fit(...))
print(model.summary())
```

```python
# Cell 2 – Save as native Keras v3 format (RECOMMENDED)
model.save('/content/dog_breed_model.keras')
print("Saved!")
```

```python
# Cell 3 – OR save as legacy HDF5 (if above fails)
model.save('/content/dog_breed_model.h5')
print("Saved!")
```

### 2. Download the saved file

```python
# Cell 4 – Download to your computer
from google.colab import files
files.download('/content/dog_breed_model.keras')   # or .h5
```

### 3. Place the file in the webapp

Copy the downloaded file into:
```
dog-breed-webapp/
└── model/
    └── dog_breed_model.keras    ← put it here
```

### 4. Update MODEL_PATH and restart

**Option A** – Rename the file to `dog_breed_model.keras` (already the default):
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Option B** – Keep your original filename, set env variable:
```powershell
$env:MODEL_PATH = "./model/my_custom_name.keras"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## ⚠️ Important: Model Output Shape

Your model must output a `(1, 120)` softmax array — one probability per breed.

Check in Colab:
```python
import numpy as np
dummy = np.zeros((1, 224, 224, 3), dtype='float32')
out = model.predict(dummy)
print("Output shape:", out.shape)  # Should be (1, 120)
```

If the shape is different (e.g. `(1, 1)` for binary or custom shapes), let me know
and I'll update the inference code accordingly.

---

## Alternative: If Your Model Variable Has a Different Name

Some notebooks use pipelines like:
```python
# Common Colab patterns:
model             # simple Sequential/Functional
history.model     # after model.fit()
loaded_model      # after tf.keras.models.load_model()
```

Run `dir()` in Colab to find it, then save with `that_variable.save(...)`.
