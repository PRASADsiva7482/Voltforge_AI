FROM python:3.12-slim-bookworm
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu -r requirements.txt
COPY api/ ./api/
COPY engine/ ./engine/
COPY config.py main.py bootstrap.py download_model.py model_manifest.json embedding_model.py embedding_manifest.json ./
RUN mkdir -p /app/model/artifacts && useradd --uid 10001 --create-home voltforge && chown -R voltforge:voltforge /app
USER voltforge
ENV VOLTFORGE_AI_HOST=0.0.0.0 VOLTFORGE_AI_ENV=production
EXPOSE 2002
# Supply VOLTFORGE_AI_API_TOKEN and mount /app/model for offline restarts.
# The pinned chat and embedding GGUFs are verified/downloaded on first startup.
CMD ["python", "main.py"]
