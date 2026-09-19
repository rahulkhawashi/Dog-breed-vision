import os
import argparse
import tensorflow as tf
import mlflow
import mlflow.tensorflow

# This script is a placeholder for the actual training logic found in dog_visionproject.ipynb.
# In a real implementation, you would load the labels.csv, prepare the image dataset,
# and build/train the model exactly as done in the notebook.

def train(epochs=1, batch_size=32):
    print("Starting MLflow tracking...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Dog_Breed_Classification")

    # Autolog metrics, parameters, and models
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        print("Building model...")
        # MobileNetV2 base
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.Dropout(0.2)(x)
        # 120 breeds for Stanford Dogs dataset
        outputs = tf.keras.layers.Dense(120, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        print(f"Training for {epochs} epochs...")
        # In a real scenario, model.fit() would be called here with actual data batches.
        # This is where the notebook logic for `create_data_batches` belongs.

        print("Finished training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch_size=args.batch_size)
