# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip  # Perbarui pip
RUN pip install --no-cache-dir -r requirements.txt


# Copy project files
COPY src/ /app/src/
COPY data/ /app/data/
COPY models/ /app/models/

# Default command to run
CMD ["python", "src/data_ingest.py"]