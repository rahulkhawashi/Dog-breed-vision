"""
model_utils.py – Model loading, image preprocessing, and inference utilities.

Supports:
  • Native Keras / TF SavedModel  (.h5, .keras, SavedModel directory)
  • Joblib + dill serialised callables (.joblib) loaded safely via subprocess
    to avoid TF-re-initialisation deadlocks in the main server process.
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import tempfile
from typing import List

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SIZE = (224, 224)   # Width × Height expected by MobileNetV2 / most CNNs
CHANNELS    = 3            # RGB


# ---------------------------------------------------------------------------
# SubprocessDillModel
# ---------------------------------------------------------------------------

# Inline helper script that runs inside a fresh Python process.
# It loads the joblib/dill object and runs a single prediction, then
# writes the result JSON to stdout and exits.  This completely avoids
# the TF-already-initialised deadlock.
_SUBPROCESS_PREDICT_SCRIPT = r"""
import sys, json
import numpy as np
import joblib, dill  # noqa: F401  (dill must be imported for joblib to use it)

model_path  = sys.argv[1]
tensor_path = sys.argv[2]   # .npy file with the preprocessed tensor

obj    = joblib.load(model_path)
tensor = np.load(tensor_path)

if hasattr(obj, "predict"):
    raw = obj.predict(tensor, verbose=0)
elif callable(obj):
    raw = obj(tensor)
else:
    raise RuntimeError(f"Loaded object ({type(obj)}) is not callable and has no .predict()")

probs = np.squeeze(np.array(raw)).tolist()
print(json.dumps(probs))
"""


class SubprocessDillModel:
    """
    Pseudo-model that runs inference in a fresh Python subprocess.

    This is used when the .joblib file contains a dill-serialised object that
    triggers a TF graph re-initialisation and deadlocks if loaded inside the
    same process as the FastAPI server (which already has TF running).

    Each call to .predict() spawns a short-lived subprocess that:
      1. Loads the joblib/dill object fresh (no conflict with existing TF).
      2. Runs a single forward pass on the supplied tensor.
      3. Returns the probabilities as JSON and exits.
    """

    def __init__(self, model_path: str):
        self._model_path = model_path
        # Write the helper script to a temp file once at construction time.
        self._script = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="dill_predict_"
        )
        self._script.write(_SUBPROCESS_PREDICT_SCRIPT)
        self._script.close()
        logger.info(
            "SubprocessDillModel initialised. Helper script: %s", self._script.name
        )

    def predict(self, tensor: np.ndarray, verbose: int = 0) -> np.ndarray:
        # Serialise the tensor to a temp .npy file
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tf_file:
            tensor_path = tf_file.name
            np.save(tf_file, tensor)

        try:
            result = subprocess.run(
                [sys.executable, self._script.name, self._model_path, tensor_path],
                capture_output=True,
                text=True,
                timeout=120,   # 2-minute hard timeout per prediction
            )
        finally:
            os.unlink(tensor_path)

        if result.returncode != 0:
            # Exit code 3221225477 == 0xC0000005 == Windows Access Violation
            # This happens when a dill-serialised Colab model is loaded on Windows
            # with a different TF version – the pickle stream references addresses
            # that no longer exist, causing a segfault.
            AV_EXIT = 3221225477  # 0xC0000005
            if result.returncode in (AV_EXIT, -11):  # -11 = SIGSEGV on Linux
                raise RuntimeError(
                    "The model file './model/dog_breed_model.joblib' was saved in "
                    "Google Colab (Linux / older TF) and cannot be loaded on this "
                    "machine (Windows / TF 2.20) — the dill pickle references "
                    "platform-specific memory that causes a crash. "
                    "Please re-export your model from Colab with:\n"
                    "    model.save('dog_breed_model.keras')\n"
                    "then copy it to the model/ folder and restart the server. "
                    "See EXPORT_MODEL_FROM_COLAB.md for full instructions."
                )
            raise RuntimeError(
                f"Subprocess inference failed (exit {result.returncode}):\n"
                f"{result.stderr[-1000:]}"
            )

        try:
            probs = json.loads(result.stdout.strip())
            return np.array(probs, dtype=np.float32)[np.newaxis, :]
        except Exception as exc:
            raise RuntimeError(
                f"Could not parse subprocess output: {result.stdout!r}"
            ) from exc

    def __del__(self):
        try:
            os.unlink(self._script.name)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_keras_model(path: str):
    """
    Load a Keras / TF model from *path*.

    Strategy:
    1. Try ``tf.keras.models.load_model`` (handles .h5, .keras, SavedModel dirs).
    2. Fall back to ``SubprocessDillModel`` for .joblib / dill-serialised objects.
       The subprocess approach prevents TF-deadlock when loading dill objects
       inside an already-running TF process.

    Returns the loaded model (or SubprocessDillModel) object.
    Raises ``RuntimeError`` if loading fails via both strategies.
    """
    import tensorflow as tf  # deferred import – TF startup is slow

    # --- Strategy 1: native Keras ---
    try:
        logger.info("Attempting tf.keras.models.load_model('%s') …", path)
        model = tf.keras.models.load_model(path)
        logger.info("Model loaded via Keras (type=%s).", type(model).__name__)
        return model
    except Exception as keras_err:
        logger.warning("Keras load failed: %s", keras_err)

    # --- Strategy 2: subprocess dill (safe joblib loading) ---
    logger.warning(
        "Keras load failed for '%s'. Falling back to SubprocessDillModel.\n"
        "NOTE: If this model was saved in Google Colab, it may crash on Windows "
        "due to a platform/TF-version mismatch. Re-export it from Colab using:\n"
        "    model.save('dog_breed_model.keras')\n"
        "See EXPORT_MODEL_FROM_COLAB.md for full step-by-step instructions.",
        path,
    )
    return SubprocessDillModel(path)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes into a normalised (1, 224, 224, 3) NumPy tensor.

    Pipeline:
      1. Decode bytes → PIL Image (raises ValueError for non-images)
      2. Convert to RGB (handles RGBA / grayscale / palette modes)
      3. Resize to 224×224 using high-quality Lanczos resampling
      4. Normalise pixel values from [0, 255] → [0.0, 1.0]
      5. Add batch dimension → shape (1, 224, 224, 3)

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        NumPy float32 array of shape (1, 224, 224, 3).

    Raises:
        ValueError: If *image_bytes* cannot be decoded as an image.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except (UnidentifiedImageError, Exception) as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    # Ensure RGB (model expects 3 channels)
    image = image.convert("RGB")

    # Resize to model's expected input size
    image = image.resize(TARGET_SIZE, resample=Image.LANCZOS)

    # Convert to float32 numpy array and normalise to [0, 1]
    arr = np.array(image, dtype=np.float32) / 255.0

    # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    tensor = np.expand_dims(arr, axis=0)

    return tensor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_breed(model, tensor: np.ndarray, top_k: int = 3) -> List[dict]:
    """
    Run inference and return the top-k predictions.

    Args:
        model:  Loaded Keras model or SubprocessDillModel with a ``predict`` method.
        tensor: Preprocessed image tensor of shape (1, 224, 224, 3).
        top_k:  Number of top predictions to return (default 3).

    Returns:
        List of dicts sorted by confidence descending, each containing:
          - ``index``      (int)   class index
          - ``confidence`` (float) softmax probability in [0, 1]
    """
    from breeds import BREED_CLASSES  # local import to avoid circular deps

    # Run forward pass – result shape: (1, num_classes)
    raw_output = model.predict(tensor, verbose=0)

    # Flatten to 1-D probability array
    probabilities: np.ndarray = np.squeeze(raw_output)

    # Guard: ensure probabilities is 1-D
    if probabilities.ndim != 1:
        raise RuntimeError(
            f"Unexpected model output shape {raw_output.shape}. "
            "Expected (1, num_classes)."
        )

    # Clamp top_k to number of available classes
    num_classes = len(probabilities)
    top_k = min(top_k, num_classes)

    # argsort ascending, then take last top_k reversed → descending order
    top_indices: np.ndarray = np.argsort(probabilities)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append(
            {
                "index": int(idx),
                "confidence": float(probabilities[idx]),
            }
        )

    return results
