"""
convert_joblib_to_keras.py
──────────────────────────
One-time conversion script: loads the dill/joblib-serialised model and
re-saves it as a native Keras .keras file that can be loaded reliably.

Run ONCE before starting the server:
    python convert_joblib_to_keras.py

This produces  model/dog_breed_model.keras  which the FastAPI server then
loads via  tf.keras.models.load_model()  without any dill/subprocess issues.
"""

import os, sys

SRC  = "./model/dog_breed_model.joblib"
DEST = "./model/dog_breed_model.keras"

if not os.path.exists(SRC):
    sys.exit(f"Source not found: {SRC}")

if os.path.exists(DEST):
    print(f"Output already exists: {DEST}")
    print("Delete it first if you want to re-convert.")
    sys.exit(0)

print(f"Loading {SRC} via joblib+dill …  (may take 1–3 minutes)")
import joblib, dill  # dill must be installed for this to work
obj = joblib.load(SRC)
print(f"Loaded object type: {type(obj)}")

# The joblib might contain:
#  a) A Keras Sequential / Functional model  → save directly
#  b) A sklearn Pipeline with a Keras step   → extract the Keras model
#  c) A raw callable (dill'd predict fn)     → cannot convert; inform user

import tensorflow as tf

keras_model = None

# Case (a): direct Keras model
if isinstance(obj, tf.keras.Model):
    keras_model = obj

# Case (b): sklearn Pipeline – look for a Keras model step
elif hasattr(obj, "steps"):
    for name, step in obj.steps:
        if isinstance(step, tf.keras.Model):
            keras_model = step
            print(f"Extracted Keras model from pipeline step '{name}'.")
            break
    if keras_model is None:
        for name, step in obj.steps:
            if hasattr(step, "model") and isinstance(step.model, tf.keras.Model):
                keras_model = step.model
                print(f"Extracted Keras model from pipeline step '{name}'.model")
                break

# Case (c): raw callable – try calling it with a dummy tensor to verify shape
elif callable(obj):
    print(
        "The joblib file contains a bare callable (dill-serialised predict function).\n"
        "This cannot be converted to .keras format.\n"
        "Please re-export your model from Colab using:\n"
        "    model.save('dog_breed_model.keras')   # Keras v3\n"
        "or:\n"
        "    model.save('dog_breed_model.h5')      # legacy HDF5\n"
        "Then copy the file to the model/ directory."
    )
    sys.exit(1)

if keras_model is None:
    print(
        f"Could not find a Keras model inside the joblib object (type={type(obj)}).\n"
        "Please re-export your model from Colab with model.save()."
    )
    sys.exit(1)

print(f"Saving Keras model to {DEST} …")
keras_model.save(DEST)
print(f"Done! Model saved to {DEST}")
print(f"Now set MODEL_PATH=./model/dog_breed_model.keras and restart the server.")
