# Build & Run stage
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install small runtime/build dependencies used by FastAPI and HTML parsing libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app.py .
COPY dataset.txt .

# Expose port 2002
EXPOSE 2002

# Run FastAPI app with Uvicorn
CMD ["python", "app.py"]
