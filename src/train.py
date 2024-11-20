import pandas as pd
import mlflow
from model import lstm_model

def train_model():
    # Set MLflow Tracking URI
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Load training data
    train_df = pd.read_csv('data/processed/train_df.csv')
    X_train = train_df.drop(columns=['y_train'])
    y_train = train_df['y_train']

    epochs = 10
    batch_size = 32

    with mlflow.start_run(run_name="Model Training"):
        model = lstm_model((X_train.shape[1], 1))
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

        # Save model locally
        model.save("models/lstm_model.h5")

        # Log parameters and metrics
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size})
        mlflow.log_metric("train_loss", history.history["loss"][-1])

        # Log model to MLflow
        mlflow.keras.log_model(model, artifact_path="model")
        mlflow.log_artifact("models/lstm_model.h5")  # Optional: Save to models/

        # Register the model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/models"
        model_name = "stock_price_model"
        registered_model = mlflow.register_model(model_uri, model_name)

        print(f"Model registered with version: {registered_model.version}")

if __name__ == "__main__":
    train_model()
