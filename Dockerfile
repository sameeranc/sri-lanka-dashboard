FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by geopandas / shapely
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces uses port 7860
EXPOSE 7860

CMD ["gunicorn", "dashboard:server", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1"]
