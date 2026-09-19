import os
import shutil
import mlflow
from mlflow.tracking import MlflowClient

def export_best_model(experiment_name="Dog_Breed_Classification", dest_path="../dog-breed-webapp/model/dog_breed_model.keras"):
    print(f"Connecting to MLflow Tracking Server...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found.")
        return

    print(f"Searching for the best run in experiment '{experiment_name}'...")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )

    if not runs:
        print("No runs found.")
        return

    best_run = runs[0]
    print(f"Best run ID: {best_run.info.run_id}")
    print(f"Best accuracy: {best_run.data.metrics.get('accuracy', 'N/A')}")

    # Fetch model URI
    model_uri = f"runs:/{best_run.info.run_id}/model"
    print(f"Model URI: {model_uri}")

    # Download model artifact
    # Note: mlflow.tensorflow logs SavedModel or Keras format. We will copy it to the dest_path.
    local_path = mlflow.artifacts.download_artifacts(run_id=best_run.info.run_id, artifact_path="model/data/model.keras")
    
    if os.path.exists(local_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(local_path, dest_path)
        print(f"Successfully exported best model to {dest_path}")
    else:
        print("Could not locate the model artifact.")

if __name__ == "__main__":
    export_best_model()
