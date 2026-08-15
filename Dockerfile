# Build & Run stage
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install runtime/build dependencies
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

# Copy complete modular source code
COPY api/ ./api/
COPY engine/ ./engine/
COPY model/ ./model/
COPY tests/ ./tests/
COPY circuit_verifier.py .
COPY web_search_engine.py .
COPY config.py .
COPY main.py .
COPY app.py .
COPY dataset.txt .

# Expose microservice port
EXPOSE 2002

# Run FastAPI app
CMD ["python", "main.py"]
