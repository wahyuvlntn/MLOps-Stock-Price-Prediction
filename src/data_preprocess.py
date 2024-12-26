import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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

def preprocess_data():
    data = pd.read_csv('data/raw/stock_price.csv', skiprows=[1])
    df = data[['Close']].dropna()

    # Normalisasi data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Fungsi untuk membuat data time series (sequences)
    def create_sequences(data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length, 0])
            y.append(data[i + sequence_length, 0])
        return np.array(X), np.array(y)

    # Parameter sequence length
    sequence_length = 60  # Menggunakan 60 hari sebelumnya untuk memprediksi hari berikutnya

    # Membagi data menjadi train (80%), validation (10%), dan test (10%)
    total_size = len(scaled_data)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)

    # Membuat data training dan testing
    train_data = scaled_data[:train_size]
    val_data = scaled_data[train_size:train_size + val_size]
    test_data = scaled_data[train_size + val_size - sequence_length:]

    X_train, y_train = create_sequences(train_data, sequence_length)
    X_val, y_val = create_sequences(val_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)

    os.makedirs("data/processed", exist_ok=True)

    # Menggabungkan X dan y ke dalam DataFrame untuk setiap set
    train_df = pd.DataFrame({
        **{f"X_train_{i+1}": X_train[:, i] for i in range(X_train.shape[1])},
        "y_train": y_train
    })

    val_df = pd.DataFrame({
        **{f"X_val_{i+1}": X_val[:, i] for i in range(X_val.shape[1])},
        "y_val": y_val
    })

    test_df = pd.DataFrame({
        **{f"X_test_{i+1}": X_test[:, i] for i in range(X_test.shape[1])},
        "y_test": y_test
    })

    # Menyimpan data hasil ke file CSV
    train_df.to_csv('data/processed/train_df.csv', index=False)
    upload_to_minio('data/processed/train_df.csv', 'data', 'train_df.csv')

    val_df.to_csv('data/processed/val_df.csv', index=False)
    upload_to_minio('data/processed/val_df.csv', 'data', 'val_df.csv')

    test_df.to_csv('data/processed/test_df.csv', index=False)
    upload_to_minio('data/processed/test_df.csv', 'data', 'test_df.csv')

if __name__ == "__main__":
    preprocess_data()
