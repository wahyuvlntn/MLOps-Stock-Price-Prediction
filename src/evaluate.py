import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import json
import os
import time

def evaluate_model():
    # Set tracking URI ke server MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    # mlflow.set_experiment("experiment_stock_price_prediction")
    # Tentukan nama eksperimen
    experiment_name = "experiment_stock_price_prediction"
    mlflow.set_experiment(experiment_name)

    # Buat eksperimen jika tidak ada
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)

    # Tunggu file sinkronisasi
    while not os.path.exists("metadata/model_version.json"):
        print(f"Menunggu proses training selesai: model_version.json")
        time.sleep(5)

    # Baca metadata model
    with open("metadata/model_version.json", "r") as f:
        model_info = json.load(f)

    if not all(key in model_info for key in ["model_name", "version"]):
        raise ValueError("File model_version.json tidak memiliki format yang benar.")

    model_name = model_info["model_name"]
    model_version = model_info["version"]

    # Load data test
    test_df = pd.read_csv('data/processed/test_df.csv')
    if 'y_test' not in test_df.columns:
        raise ValueError("Kolom 'y_test' tidak ditemukan dalam file test_df.csv.")
    X_test = test_df.drop(columns=['y_test']).values
    y_test = test_df['y_test'].values

    # Jika model adalah LSTM, reshape data test
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Load model dari MLflow Model Registry
    model_uri = f"models:/{model_name}/{model_version}"
    print(f"Mencoba memuat model dari URI: {model_uri}...")
    model = mlflow.keras.load_model(model_uri)
    print(f"Model '{model_name}' versi {model_version} berhasil dimuat.")

    # Prediksi data test
    predictions = model.predict(X_test)

    # Hitung metrik evaluasi
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

    # Log metrik ke MLflow
    with mlflow.start_run(run_name="Model Evaluation"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_version", model_version)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)

if __name__ == "__main__":
    evaluate_model()
