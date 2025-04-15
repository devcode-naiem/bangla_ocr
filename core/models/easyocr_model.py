"""
EasyOCR model implementation for Bangla Handwriting Recognition.
"""

import os
import logging
import sys
import subprocess
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path

from .base import OCRModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import easyocr with error handling
try:
    import easyocr
except ImportError:
    logger.error("easyocr not installed. Install with: pip install easyocr")
    easyocr = None


class EasyOCRModel(OCRModel):
    """
    Implementation of OCR model using EasyOCR.
    
    This class provides methods to:
    1. Check if EasyOCR is available
    2. Get supported languages
    3. Recognize text from images using EasyOCR
    """
    
    def __init__(
        self, 
        language: str = 'bn',
        use_gpu: bool = False,
        download_enabled: bool = True,
        **kwargs
    ):
        """
        Initialize the EasyOCR model.
        
        Args:
            language: Default language for recognition ('bn' for Bengali)
            use_gpu: Whether to use GPU for recognition if available
            download_enabled: Whether to allow downloading models
            **kwargs: Additional parameters for EasyOCR
        """
        self.language = language
        self.use_gpu = use_gpu
        self.download_enabled = download_enabled
        
        # Check if easyocr is available
        if easyocr is None:
            logger.error("easyocr module not found. EasyOCR model will not be available.")
            self._available = False
            self._version = None
            self._reader = None
            self._languages = []
            return
        
        # Get EasyOCR version
        try:
            self._version = easyocr.__version__
            logger.info(f"EasyOCR version: {self._version}")
        except:
            self._version = "unknown"
            logger.warning("Could not determine EasyOCR version")
        
        # Initialize reader with the specified language
        try:
            # Get all available languages
            self._languages = self._get_available_languages()
            
            # Check if requested language is supported
            if self.language not in self._languages:
                logger.warning(f"Language '{language}' not available in EasyOCR")
                if 'bn' in self._languages:
                    logger.info("Using 'bn' (Bengali) as fallback language")
                    self.language = 'bn'
                elif len(self._languages) > 0:
                    logger.info(f"Using '{self._languages[0]}' as fallback language")
                    self.language = self._languages[0]
                else:
                    logger.warning("No languages available in EasyOCR")
                    self.language = None
                    self._available = False
                    self._reader = None
                    return
            
            # Initialize the reader
            logger.info(f"Initializing EasyOCR with language '{self.language}' (GPU: {self.use_gpu})")
            
            # Set download flag based on preference
            if not self.download_enabled:
                os.environ['EASYOCR_DOWNLOAD_ENABLED'] = '0'
            
            # Create reader with requested language
            self._reader = easyocr.Reader(
                [self.language], 
                gpu=self.use_gpu,
                **kwargs
            )
            
            self._available = True
            logger.info("EasyOCR initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self._available = False
            self._reader = None
    
    def _get_available_languages(self) -> List[str]:
        """
        Get list of languages supported by EasyOCR.
        
        Returns:
            List of language codes
        """
        try:
            # EasyOCR available languages
            available_languages = easyocr.utils.get_available_languages()
            return available_languages
        except Exception as e:
            logger.error(f"Error getting EasyOCR languages: {e}")
            return []
    
    def is_available(self) -> bool:
        """
        Check if EasyOCR is available.
        
        Returns:
            Boolean indicating if EasyOCR is available
        """
        return self._available and self._reader is not None
    
    def get_available_languages(self) -> List[str]:
        """
        Get list of available languages.
        
        Returns:
            List of language codes
        """
        return self._languages
    
    def recognize(
        self, 
        image: np.ndarray, 
        detail: int = 0,
        paragraph: bool = False,
        min_size: int = 20,
        **kwargs
    ) -> Tuple[str, float]:
        """
        Recognize text from an image using EasyOCR.
        
        Args:
            image: Input image as numpy array
            detail: Level of detail in the result (0: text only, 1: boxes and text)
            paragraph: Whether to group text into paragraphs
            min_size: Minimum text box size to consider
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (recognized text, confidence score)
        """
        if not self._available or self._reader is None:
            logger.error("EasyOCR is not available")
            return "", 0.0
        
        try:
            # Call EasyOCR readtext method
            result = self._reader.readtext(
                image,
                detail=1,  # Always get details for confidence
                paragraph=paragraph,
                # Add min_size only if provided and > 0
                **({"min_size": min_size} if min_size > 0 else {}),
                **kwargs
            )
            
            # Extract text and confidence
            texts = []
            confidences = []
            
            for bbox, text, conf in result:
                texts.append(text)
                confidences.append(conf)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Join all text blocks
            if paragraph:
                # Try to maintain paragraph structure
                full_text = '\n'.join(texts)
            else:
                # Simple space-separated text
                full_text = ' '.join(texts)
            
            # Post-process the text
            full_text = self._postprocess_text(full_text)
            
            return full_text, avg_confidence
            
        except Exception as e:
            logger.error(f"Error recognizing text with EasyOCR: {e}")
            return "", 0.0
    
    def _postprocess_text(self, text: str) -> str:
        """
        Post-process recognized text.
        
        Args:
            text: Raw recognized text
            
        Returns:
            Processed text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = [' '.join(line.split()) for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # TODO: Add Bangla-specific post-processing if needed
        
        return text
    
    def get_name(self) -> str:
        """
        Get the name of this OCR model.
        
        Returns:
            Model name as string
        """
        return "EasyOCR"
    
    def get_version(self) -> Optional[str]:
        """
        Get the version of EasyOCR.
        
        Returns:
            Version string or None if not available
        """
        return self._version
    
    @classmethod
    def install_if_needed(cls) -> bool:
        """
        Install EasyOCR if not already installed.
        
        Returns:
            Boolean indicating success
        """
        if easyocr is not None:
            logger.info("EasyOCR is already installed")
            return True
        
        try:
            logger.info("Attempting to install EasyOCR...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "easyocr"])
            logger.info("EasyOCR installed successfully")
            
            # Re-import after installation
            global easyocr
            import easyocr
            return True
        except Exception as e:
            logger.error(f"Failed to install EasyOCR: {e}")
            return False
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get mapping of language codes to names.
        
        Returns:
            Dictionary mapping language codes to names
        """
        # Common language mapping for EasyOCR
        language_map = {
            'bn': 'Bengali',
            'en': 'English',
            'hi': 'Hindi',
            'ar': 'Arabic',
            'ta': 'Tamil',
            'ur': 'Urdu',
            'sa': 'Sanskrit',
            'fa': 'Persian',
            'ne': 'Nepali',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'th': 'Thai',
            'ru': 'Russian'
        }
        
        # Filter to only available languages
        available_langs = {}
        for code in self._languages:
            if code in language_map:
                available_langs[code] = language_map[code]
            else:
                available_langs[code] = code
        
        return available_langs