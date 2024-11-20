import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import mlflow
from train import train_model
# Parameter retraining
THRESHOLD_MSE = 0.02  # Tentukan ambang batas MSE sesuai kebutuhan

def load_latest_model():
    model_uri = "models:/bbca_lstm_model/Production"
    return mlflow.keras.load_model(model_uri)

def load_new_data(filepath="data/processed/train_df.csv"):
    # Load data terbaru yang belum dilihat model
    train_df = pd.read_csv(filepath)
    return train_df

def evaluate_and_retrain():
    # Load model dan data terbaru
    model = load_latest_model()
    train_df = load_new_data()
    X_train = train_df.drop(columns=['y_train'])
    y_train = train_df['y_train']


    # Prediksi dengan model
    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    print(f"Mean Squared Error on new data: {mse}")

    # Cek apakah MSE melebihi ambang batas
    if mse > THRESHOLD_MSE:
        print("Model performance decreased, triggering retraining...")
        train_model()  # Panggil fungsi untuk retraining
    else:
        print("Model performance is acceptable. No retraining required.")

if __name__ == "__main__":
    evaluate_and_retrain()
