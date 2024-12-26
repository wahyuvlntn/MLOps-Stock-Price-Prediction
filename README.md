# README

## Pendahuluan
Proyek ini adalah pipeline pembelajaran mesin untuk prediksi harga saham menggunakan model LSTM. Sistem ini mencakup proses mulai dari pengambilan data (data ingestion), preprocessing, pelatihan model, evaluasi model, hingga pemantauan kinerja model.

## Prasyarat
Pastikan Anda memiliki perangkat berikut terinstal:

1. Docker dan Docker Compose
2. Python 3.8 atau lebih baru (jika menjalankan di luar Docker)
3. Git (opsional, jika Anda mengkloning repositori ini)

## Struktur Proyek
- `src/data_ingest.py`: Mengambil data harga saham menggunakan API Yahoo Finance.
- `src/data_preprocess.py`: Melakukan preprocessing pada data.
- `src/train.py`: Melatih model LSTM dan mengunggah ke MinIO serta MLflow.
- `src/evaluate.py`: Mengevaluasi model yang sudah dilatih.
- `src/monitoring.py`: Memantau kinerja model dan melakukan retraining jika diperlukan.
- `dvc.yaml`: Konfigurasi pipeline DVC untuk data versioning.
- `docker-compose.yml`: Konfigurasi layanan untuk pipeline ini.
- `Dockerfile`: Konfigurasi Docker untuk komponen umum.
- `Dockerfile.mlflow`: Konfigurasi Docker untuk MLflow.
- `Dockerfile.mlflow_api`: Konfigurasi Docker untuk API MLflow.

## Menjalankan Proyek

### 1. Clone Repositori
```bash
git clone <https://github.com/wahyuvlntn/MLOps-Stock-Price-Prediction>
cd <MLOps-Stock-Price-Prediction>
```

### 2. Menjalankan Docker Compose
Jalankan semua layanan menggunakan `docker-compose`:

```bash
docker-compose up --build
```

Layanan yang akan dijalankan:
- MLflow: Akses di `http://localhost:5001`
- MinIO: Akses di `http://localhost:9001`
- MLflow API: Akses di `http://localhost:5002`
- Grafana: Akses di `http://localhost:3000`

### 3. Menjalankan Pipeline Secara Manual

#### a. Data Ingestion
Menjalankan skrip untuk mengambil data saham:
```bash
docker-compose run data-ingest
```

#### b. Data Preprocessing
Melakukan preprocessing data:
```bash
docker-compose run data-prep
```

#### c. Training Model
Melatih model dengan data yang telah diproses:
```bash
docker-compose run train-model
```

#### d. Evaluasi Model
Melakukan evaluasi model:
```bash
docker-compose run evaluate-model
```

#### e. Monitoring dan Retraining
Memantau kinerja model dan retraining jika diperlukan:
```bash
docker-compose run monitoring
```

### 4. Memantau Eksperimen di MLflow
- Buka `http://localhost:5001` untuk memantau eksperimen MLflow.

### 5. Menambahkan Data ke MinIO
Anda dapat mengunggah data atau model secara manual ke MinIO melalui antarmuka web di `http://localhost:9001`. Gunakan kredensial berikut:
- **Username**: `minioadmin`
- **Password**: `minioadmin`

### 6. Menggunakan API MLflow
Endpoint untuk mengambil metrik terakhir tersedia di `http://localhost:5002/metrics`.

## Konfigurasi Tambahan

### DVC Pipeline
Pipeline DVC dapat dijalankan secara manual:
```bash
dvc repro
```
Pastikan DVC sudah diinstal di sistem Anda.

### Logging
Semua log proses tersedia di folder `logs` dalam masing-masing kontainer Docker.

## Troubleshooting
- Jika ada error terkait `metadata`, pastikan folder `metadata` sudah tersedia di root proyek.
- Jika layanan Docker tidak berjalan, pastikan port yang dibutuhkan tidak digunakan oleh layanan lain.


