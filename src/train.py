import pandas as pd
import mlflow
import json
import os
from model import lstm_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

def upload_to_minio(file_path, bucket_name, object_name):
    from minio import Minio
    from minio.error import S3Error

    # Inisialisasi client MinIO
    minio_client = Minio(
        "minio:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
    )

    # Cek apakah bucket sudah ada, jika tidak buat bucket
    if not minio_client.bucket_exists(bucket_name):
        print(f"Bucket {bucket_name} tidak ditemukan, membuat bucket...")
        minio_client.make_bucket(bucket_name)

    # Mengunggah file ke bucket MinIO
    try:
        minio_client.fput_object(bucket_name, object_name, file_path)
        print(f"File {file_path} berhasil diunggah ke bucket {bucket_name} sebagai {object_name}.")
    except S3Error as e:
        print(f"Error saat mengunggah ke MinIO: {e}")

def train_model():
    # Set MLflow Tracking URI
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("experiment_stock_price_prediction")

    train_df = pd.read_csv('data/processed/train_df.csv')
    X_train = train_df.drop(columns=['y_train'])
    y_train = train_df['y_train']

    val_df = pd.read_csv('data/processed/val_df.csv')
    X_val = val_df.drop(columns=['y_val'])
    y_val = val_df['y_val']

    epochs = 20
    batch_size = 32

    with mlflow.start_run(run_name="Model Training"):
        model = lstm_model((X_train.shape[1], 1))
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

        model_path = "models/lstm_model.h5"
        model.save(model_path)

        upload_to_minio(model_path, "models", "lstm_model.h5")

        mlflow.log_params({"epochs": epochs, "batch_size": batch_size})
        mlflow.log_metric("train_loss", history.history["loss"][-1])
        mlflow.log_metric("val_loss", history.history["val_loss"][-1])

        y_val_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_val_pred)
        mae = mean_absolute_error(y_val, y_val_pred)

        mlflow.log_metric("val_mse", mse)
        mlflow.log_metric("val_mae", mae)

        # Log model MLflow
        mlflow.keras.log_model(model, artifact_path="models")
        mlflow.log_artifact("models/lstm_model.h5")  # Optional: Save to models/

        # Register model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/models"
        model_name = "stock_price_model"
        registered_model = mlflow.register_model(model_uri, model_name)

        # Save model version to a JSON file
        model_version_info = {"model_name": model_name, "version": registered_model.version}
        os.makedirs("metadata", exist_ok=True)  
        with open("metadata/model_version.json", "w") as f:
            json.dump(model_version_info, f)

        print(f"Model registered with version: {registered_model.version}")

        sync_file = "metadata/train_complete.txt"
        with open(sync_file, "w") as f:
            f.write("Training and model upload completed.")


if __name__ == "__main__":
    train_model()
