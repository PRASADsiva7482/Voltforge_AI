import logging
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATASET_PATH = os.path.join(BASE_DIR, "dataset.txt")
DATASHEET_CACHE_PATH = os.path.join(BASE_DIR, "datasheet_cache.json")
MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, "model", "artifacts")

HOST = os.getenv("VOLTFORGE_AI_HOST", "127.0.0.1")
PORT = int(os.getenv("VOLTFORGE_AI_PORT", "2002"))

CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

# Database Configuration (matches VoltForge backend setup)
DB_HOST = os.getenv("VOLTFORGE_DB_HOST", "100.73.197.23")
DB_PORT = int(os.getenv("VOLTFORGE_DB_PORT", "3306"))
DB_USER = os.getenv("VOLTFORGE_DB_USERNAME", os.getenv("VOLTFORGE_DB_USER", "root"))
DB_PASSWORD = os.getenv("VOLTFORGE_DB_PASSWORD", "SIva@7482")
DB_NAME = os.getenv("VOLTFORGE_AI_DB_NAME", "voltforge_ai")
DB_ENABLED = os.getenv("VOLTFORGE_AI_DB_ENABLED", "true").lower() in ("true", "1", "yes")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("voltforge-ai")

logger.info(f"Voltforge AI configuration loaded. Base dir: {BASE_DIR}, DB: {DB_HOST}:{DB_PORT}/{DB_NAME}")
