import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow

def evaluate_model():
    # Set tracking URI ke server MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Load data test
    test_df = pd.read_csv('data/processed/test_df.csv')
    X_test = test_df.drop(columns=['y_test']).values
    y_test = test_df['y_test'].values

    # Load scaler jika digunakan sebelumnya
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_test_scaled = scaler.fit_transform(X_test)  # Gunakan scaler yang sama seperti training

    # Reshape data untuk LSTM
    X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

    # Load model dari MLflow Model Registry
    model_uri = "models:/stock_price_model/1"  # Gunakan versi atau stage yang sesuai
    model = mlflow.keras.load_model(model_uri)

    # Prediksi data test
    predictions = model.predict(X_test_scaled)

    # Hitung metrik evaluasi
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")

    # Log metrik ke MLflow
    with mlflow.start_run(run_name="Model Evaluation"):
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

if __name__ == "__main__":
    evaluate_model()
