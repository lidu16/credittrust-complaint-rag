import os
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent.parent

# Data paths
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FILTERED_CSV_PATH = DATA_PROCESSED_DIR / "filtered_complaints.csv"

# Vector store
VECTOR_STORE_PATH = ROOT_DIR / "vector_store"

# Model settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# LLM settings
LLM_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
