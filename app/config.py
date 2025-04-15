"""
Configuration settings for the Bangla Handwriting Recognition System
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULT_DIR = STATIC_DIR / "results"

# Create directories if they don't exist
STATIC_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_TITLE = "Bangla Handwriting Recognition API"
API_DESCRIPTION = "API for recognizing Bangla handwritten text from images"
API_VERSION = "0.1.0"

# OCR settings
TESSERACT_PATH = os.getenv("TESSERACT_PATH", None)  # Use system default if not specified
TESSDATA_PREFIX = os.getenv("TESSDATA_PREFIX", None)
if TESSDATA_PREFIX:
    os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX

# Default OCR settings
DEFAULT_OCR_ENGINE = os.getenv("DEFAULT_OCR_ENGINE", "easyocr")  # Options: tesseract, easyocr, both
DEFAULT_PREPROCESS_METHOD = os.getenv("DEFAULT_PREPROCESS_METHOD", "auto")  # Options: auto, adaptive, otsu, local, denoise

# Performance settings
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1920"))  # Max width/height in pixels
EASYOCR_GPU = os.getenv("EASYOCR_GPU", "False").lower() in ["true", "1", "yes"]

# Caching settings
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "True").lower() in ["true", "1", "yes"]
CACHE_EXPIRATION = int(os.getenv("CACHE_EXPIRATION", "3600"))  # In seconds (1 hour default)