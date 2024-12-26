import pandas as pd
import json
import os
from tensorflow.keras.models import load_model
from minio import Minio
from minio.error import S3Error
import mlflow.keras
from sklearn.metrics import mean_squared_error
import numpy as np
import time

def download_model_from_minio(bucket_name, object_name, destination):
    """Download the model from MinIO."""
    minio_client = Minio(
        "minio:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
    )

    if not minio_client.bucket_exists(bucket_name):
        raise Exception(f"Bucket {bucket_name} does not exist.")

    try:
        minio_client.fget_object(bucket_name, object_name, destination)
        print(f"Model {object_name} downloaded from MinIO bucket {bucket_name}.")
    except S3Error as e:
        print(f"Error downloading model from MinIO: {e}")
        raise

def load_test_data():
    """Load test data."""
    test_data_path = 'data/processed/test_df.csv'
    if not os.path.exists(test_data_path):
        raise FileNotFoundError("Test data not found. Ensure preprocessing is complete.")

    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop(columns=['y_test']).values
    y_test = test_df['y_test'].values
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data and return the error."""
    y_pred = model.predict(X_test)
    error = mean_squared_error(y_test, y_pred)
    print(f"Test error (MSE): {error}")
    return error

def retrain_model():
    """Placeholder function to retrain the model."""
    from train import train_model
    print("Retraining the model...")
    train_model()
    print("Retraining completed.")

def upload_metadata_to_minio(file_path, bucket_name, object_name):
    """Upload metadata to MinIO."""
    minio_client = Minio(
        "minio:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
    )

    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    try:
        minio_client.fput_object(bucket_name, object_name, file_path)
        print(f"Metadata {file_path} uploaded to MinIO bucket {bucket_name} as {object_name}.")
    except S3Error as e:
        print(f"Error uploading metadata to MinIO: {e}")

if __name__ == "__main__":

    sync_file = "metadata/train_complete.txt"

    while not os.path.exists(sync_file):
        print(f"Menunggu proses training selesai: {sync_file}")
        time.sleep(5)

    print("Proses training selesai. Memulai monitoring...")

    ERROR_THRESHOLD = 0.05  # Define acceptable test error threshold

    # Download the latest model from MinIO
    model_path = "models/lstm_model.h5"
    download_model_from_minio("models", "lstm_model.h5", model_path)

    # Load the model
    # model = mlflow.keras.load_model(model_path)
    model = load_model(model_path)

    # Load test data
    X_test, y_test = load_test_data()

    # Evaluate the model
    test_error = evaluate_model(model, X_test, y_test)

    # Check if retraining is necessary
    if test_error > ERROR_THRESHOLD:
        print("Test error exceeds threshold. Initiating retraining...")
        retrain_model()

        # Upload updated metadata to MinIO
        metadata_path = "metadata/model_version.json"
        upload_metadata_to_minio(metadata_path, "metadata", "model_version.json")
    else:
        print("Test error within acceptable range. No retraining required.")
