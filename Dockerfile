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
COPY api_contract/ ./api_contract/
COPY engine/ ./engine/
COPY model/ ./model/
COPY context_compiler/ ./context_compiler/
COPY data_governance/ ./data_governance/
COPY electronics_corpus/ ./electronics_corpus/
COPY engineering_tools/ ./engineering_tools/
COPY evaluation/ ./evaluation/
COPY feedback_governance/ ./feedback_governance/
COPY grounding/ ./grounding/
COPY memory_store/ ./memory_store/
COPY internet_retrieval/ ./internet_retrieval/
COPY local_retrieval/ ./local_retrieval/
COPY task_schema/ ./task_schema/
COPY hardware_coverage/ ./hardware_coverage/
COPY tests/ ./tests/
COPY circuit_verifier.py .
COPY web_search_engine.py .
COPY observability.py .
COPY config.py .
COPY main.py .
COPY app.py .
COPY dataset.txt .

# Expose microservice port
EXPOSE 2002

# Containers need to listen on their network interface; local development
# defaults to 127.0.0.1 in config.py.
ENV VOLTFORGE_AI_HOST=0.0.0.0

# Run FastAPI app
CMD ["python", "main.py"]
