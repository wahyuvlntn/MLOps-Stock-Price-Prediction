import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import os


def ingest_data():
    # Definisikan ticker untuk saham BBCA
    ticker = "BBCA.JK"

    # Mendapatkan tanggal hari ini
    today = datetime.date.today()

    # Mendapatkan tanggal 10 tahun yang lalu
    ten_years_ago = today - datetime.timedelta(days=10*365)

    # Mengunduh data harga saham BBCA selama 10 tahun terakhir
    data = yf.download(ticker, start=ten_years_ago, end=today)

    os.makedirs("data/raw", exist_ok=True)

    data.to_csv('data/raw/stock_price.csv', index=False)


if __name__ == "__main__":
    ingest_data()