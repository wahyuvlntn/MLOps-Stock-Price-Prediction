import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess_data():
    # Membaca data harga tutup (Close) saja
    # data = pd.read_csv('data/raw/stock_price.csv')
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

    # Membuat data training dan testing
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - sequence_length:]

    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)

    os.makedirs("data/processed", exist_ok=True)

    # Menggabungkan X dan y ke dalam DataFrame untuk setiap set
    train_df = pd.DataFrame({
        **{f"X_train_{i+1}": X_train[:, i] for i in range(X_train.shape[1])},
        "y_train": y_train
    })

    test_df = pd.DataFrame({
        **{f"X_test_{i+1}": X_test[:, i] for i in range(X_test.shape[1])},
        "y_test": y_test
    })

    # Menyimpan data hasil ke file CSV
    train_df.to_csv('data/processed/train_df.csv', index=False)
    test_df.to_csv('data/processed/test_df.csv', index=False)


if __name__ == "__main__":
    preprocess_data()
