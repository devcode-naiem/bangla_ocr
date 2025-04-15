"""
OCR model implementations for Bangla Handwriting Recognition.

This package contains:
- Abstract base class for OCR models
- Tesseract OCR implementation
- EasyOCR implementation
"""

from .base import OCRModel
from .tesseract_model import TesseractModel
from .easyocr_model import EasyOCRModel

__all__ = ['OCRModel', 'TesseractModel', 'EasyOCRModel']