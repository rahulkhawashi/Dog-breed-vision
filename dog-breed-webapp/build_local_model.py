"""
build_local_model.py
─────────────────────
Builds a MobileNetV2-based dog breed classifier (120 Stanford Dogs classes)
using ImageNet pretrained weights and saves it as model/dog_breed_model.keras.

Instead of retraining or randomly initializing the classification head, this script
reuses the original ImageNet classifier and maps the 1000 ImageNet classes 
down to our 120 Stanford Dogs breeds.

Run once:
    python build_local_model.py

Then restart the server – it will load the .keras file natively.
"""

import os
import sys
import numpy as np

DEST = "./model/dog_breed_model.keras"

print("Importing TensorFlow …")
import tensorflow as tf
print(f"TensorFlow {tf.__version__}")

# ---------------------------------------------------------------------------
# Stanford Dogs 120 breeds mapping to ImageNet indices
# ---------------------------------------------------------------------------
from breeds import BREED_CLASSES

# These are the ImageNet class indices corresponding exactly to the 120 breeds in breeds.py
IMAGENET_DOG_INDICES = [
    252, 160, 275, 191, 180, 240, 193, 253, 161, 162, 181, 239, 165, 156, 163, 164, 
    232, 182, 169, 195, 233, 242, 262, 226, 215, 243, 192, 264, 209, 151, 260, 216, 
    219, 231, 206, 194, 274, 273, 236, 167, 212, 217, 241, 248, 205, 245, 235, 210, 
    197, 207, 214, 246, 257, 238, 224, 173, 213, 184, 221, 170, 171, 152, 261, 227, 
    183, 228, 222, 208, 189, 255, 204, 249, 225, 153, 268, 237, 266, 196, 256, 185, 
    174, 186, 229, 175, 157, 154, 263, 259, 254, 168, 159, 234, 247, 176, 258, 223, 
    199, 177, 190, 230, 155, 250, 201, 202, 179, 267, 198, 220, 244, 200, 265, 158, 
    211, 166, 178, 218, 203, 172, 188, 187
]

NUM_CLASSES = len(BREED_CLASSES)
print(f"Number of classes: {NUM_CLASSES}")

# ---------------------------------------------------------------------------
# Build the mapping model
# ---------------------------------------------------------------------------
print("Building mapped MobileNetV2 model …")

# The webapp preprocessing scales image pixels to [0, 1]
inputs = tf.keras.Input(shape=(224, 224, 3), name="image_input")

# MobileNetV2 expects [-1, 1], so we map [0, 1] -> [-1, 1] internally!
x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0, name="preprocess_to_neg1_pos1")(inputs)

# Use base MobileNetV2 WITH the ImageNet classification head, but return logits
try:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=True,
        weights="imagenet",
        classifier_activation=None # Get raw logits
    )
except Exception:
    # Fallback if older TF doesn't support classifier_activation=None
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=True,
        weights="imagenet"
    )

base.trainable = False

x = base(x, training=False)

# Instead of a Lambda layer, we use a Dense layer with a fixed permutation matrix
# to map the 1000 ImageNet outputs to our 120 dog breeds.
# This avoids serialization issues with Lambda layers in Keras 3.

# Create permutation matrix of shape (1000, 120)
perm_matrix = np.zeros((1000, 120), dtype=np.float32)
for j, idx in enumerate(IMAGENET_DOG_INDICES):
    perm_matrix[idx, j] = 1.0

# Apply the permutation via a Dense layer without bias
x = tf.keras.layers.Dense(
    120, 
    use_bias=False, 
    trainable=False, 
    name="gather_dog_breeds"
)(x)

# Re-apply softmax over only the 120 dog breeds
outputs = tf.keras.layers.Activation("softmax", name="predictions")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="dog_breed_mobilenetv2_mapped")

# Set the weights of the gather layer!
model.get_layer("gather_dog_breeds").set_weights([perm_matrix])

model.summary()

# ---------------------------------------------------------------------------
# Save as .keras (native format)
# ---------------------------------------------------------------------------
if os.path.exists(DEST):
    os.remove(DEST)
    
print(f"\nSaving model to {DEST} …")
model.save(DEST)
print(f"Done! Model saved to {DEST}")
print(f"\nNow restart the server. MODEL_PATH is already set to {DEST} via .env or the default.")
print("Upload a dog image to verify inference works!")
