from flask import Flask, jsonify
import mlflow
import mlflow.tracking

app = Flask(__name__)

# Configure MLflow tracking URI
MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Fetch the latest metrics from MLflow for a specific experiment.
    """
    try:
        client = mlflow.tracking.MlflowClient()

        # Replace "default" with the actual experiment name you want to fetch
        experiment_name = "experiment_stock_price_prediction"
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            return jsonify({"error": f"Experiment '{experiment_name}' not found"}), 404

        experiment_id = experiment.experiment_id

        # Get the latest run for the specified experiment
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            max_results=1,
            order_by=["attributes.start_time DESC"]
        )

        if not runs:
            return jsonify({"error": f"No runs found for experiment '{experiment_name}'"}), 404

        latest_run = runs[0]
        run_data = latest_run.data

        # Format metrics to return as JSON
        metrics = {
            "experiment_name": experiment_name,
            "experiment_id": experiment_id,
            "run_id": latest_run.info.run_id,
            "params": run_data.params,
            "metrics": run_data.metrics,
        }

        return jsonify(metrics)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
