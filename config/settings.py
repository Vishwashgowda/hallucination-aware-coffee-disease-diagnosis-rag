"""
Configuration settings for Coffee Disease Diagnosis System
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
VECTOR_DB_PATH = DATA_DIR / "vector_db"
CONFIG_DIR = PROJECT_ROOT / "config"

# PDF Loader settings
PDF_CHUNK_SIZE = 500
PDF_CHUNK_OVERLAP = 50

# Vector Store settings
VECTOR_DB_TYPE = "faiss"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Retriever settings
RETRIEVER_TOP_K = 5

# Diagnosis settings
MIN_CONFIDENCE_THRESHOLD = 0.8
MAX_QUESTIONS = 3
RELEVANCE_THRESHOLD = 0.3

# Hallucination checker settings
NUM_GENERATIONS_FOR_VERIFICATION = 3

# API settings
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "coffee_diagnosis.log"
