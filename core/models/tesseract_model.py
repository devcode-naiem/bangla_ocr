"""
Tesseract OCR model implementation for Bangla Handwriting Recognition.
"""

import os
import logging
import subprocess
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path

from .base import OCRModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pytesseract with error handling
try:
    import pytesseract
except ImportError:
    logger.error("pytesseract not installed. Install with: pip install pytesseract")
    pytesseract = None


class TesseractModel(OCRModel):
    """
    Implementation of OCR model using Tesseract.
    
    This class provides methods to:
    1. Check if Tesseract is available
    2. Get supported languages
    3. Recognize text from images using Tesseract OCR
    """
    
    def __init__(
        self, 
        tesseract_path: Optional[str] = None,
        language: str = 'ben',
        **kwargs
    ):
        """
        Initialize the Tesseract OCR model.
        
        Args:
            tesseract_path: Path to Tesseract executable
            language: Default language for recognition
            **kwargs: Additional parameters
        """
        self.language = language
        
        # Check if pytesseract is available
        if pytesseract is None:
            logger.error("pytesseract module not found. Tesseract model will not be available.")
            self._available = False
            return
        
        # Set Tesseract path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            logger.info(f"Using Tesseract executable at: {tesseract_path}")
        
        # Check if Tesseract is available
        try:
            version = pytesseract.get_tesseract_version()
            self._version = str(version)
            logger.info(f"Tesseract version: {version}")
            self._available = True
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract: {e}")
            self._available = False
            self._version = None
        
        # Get available languages
        if self._available:
            self._languages = self._get_languages()
            if self.language not in self._languages:
                logger.warning(f"Requested language '{language}' not available in Tesseract")
                if 'ben' in self._languages:
                    logger.info("Using 'ben' (Bengali) as fallback language")
                    self.language = 'ben'
                elif len(self._languages) > 0 and 'eng' in self._languages:
                    logger.info("Using 'eng' as fallback language")
                    self.language = 'eng'
                elif len(self._languages) > 0:
                    logger.info(f"Using '{self._languages[0]}' as fallback language")
                    self.language = self._languages[0]
                else:
                    logger.warning("No languages available in Tesseract")
                    self.language = None
                    self._available = False
    
    def _get_languages(self) -> List[str]:
        """
        Get list of languages supported by Tesseract.
        
        Returns:
            List of language codes
        """
        try:
            result = subprocess.run(
                [pytesseract.pytesseract.tesseract_cmd, '--list-langs'],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                logger.warning(f"Error getting languages: {result.stderr}")
                return []
            
            # Parse languages from output
            langs = result.stdout.strip().split('\n')
            if langs and langs[0].startswith('List of'):
                langs = langs[1:]
            
            return langs
        except Exception as e:
            logger.error(f"Error getting languages: {e}")
            return []
    
    def is_available(self) -> bool:
        """
        Check if Tesseract is available.
        
        Returns:
            Boolean indicating if Tesseract is available
        """
        return self._available
    
    def get_available_languages(self) -> List[str]:
        """
        Get list of available languages.
        
        Returns:
            List of language codes
        """
        if not self._available:
            return []
        
        return self._languages
    
    def recognize(
        self, 
        image: np.ndarray, 
        language: Optional[str] = None,
        config: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, float]:
        """
        Recognize text from an image using Tesseract.
        
        Args:
            image: Input image as numpy array
            language: Language to use for recognition (overrides default)
            config: Tesseract configuration string
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (recognized text, confidence score)
        """
        if not self._available:
            logger.error("Tesseract is not available")
            return "", 0.0
        
        # Use provided language or default
        lang = language if language else self.language
        
        # Check if language is available
        if lang and lang not in self._languages:
            logger.warning(f"Language '{lang}' not available in Tesseract")
            if 'ben' in self._languages:
                logger.info("Falling back to 'ben' (Bengali)")
                lang = 'ben'
            elif len(self._languages) > 0:
                logger.info(f"Falling back to '{self._languages[0]}'")
                lang = self._languages[0]
            else:
                logger.warning("No languages available, using Tesseract default")
                lang = None
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Default configs optimized for Bangla handwriting
        configs = kwargs.get('configs', [
            '--psm 6 --oem 3',  # Assume single block of text, LSTM only
            '--psm 4 --oem 3',  # Assume single column, LSTM only
            '--psm 3 --oem 3',  # Auto page segmentation, LSTM only
            '--psm 11 --oem 3', # Sparse text, LSTM only
            '--psm 13 --oem 3'  # Raw line, LSTM only
        ])
        
        # If specific config provided, use only that
        if config:
            configs = [config]
        
        best_text = ""
        best_score = 0.0
        
        # Try multiple configs
        for cfg in configs:
            try:
                # Build command with language if available
                cmd = f"{cfg}"
                if lang:
                    cmd += f" -l {lang}"
                
                # Get both text and data for confidence score
                text = pytesseract.image_to_string(image, config=cmd)
                data = pytesseract.image_to_data(image, config=cmd, output_type=pytesseract.Output.DICT)
                
                # Calculate confidence score (average of word confidences)
                if 'conf' in data and data['conf']:
                    confidences = [c for c in data['conf'] if c != -1]  # Filter invalid confidences
                    if confidences:
                        score = sum(confidences) / len(confidences) / 100.0  # Normalize to 0-1
                    else:
                        score = 0
                else:
                    score = 0
                
                # Update best result if better
                if len(text.strip()) > len(best_text.strip()) or (
                    len(text.strip()) == len(best_text.strip()) and score > best_score):
                    best_text = text
                    best_score = score
                    
            except Exception as e:
                logger.warning(f"Error with Tesseract config {cfg}: {e}")
        
        # Post-process text
        best_text = self._postprocess_text(best_text)
        
        return best_text, best_score
    
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
        text = ' '.join(text.split())
        
        # TODO: Add Bangla-specific post-processing if needed
        
        return text
    
    def get_name(self) -> str:
        """
        Get the name of this OCR model.
        
        Returns:
            Model name as string
        """
        return "Tesseract OCR"
    
    def get_version(self) -> Optional[str]:
        """
        Get the version of Tesseract.
        
        Returns:
            Version string or None if not available
        """
        return self._version
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of this Tesseract implementation.
        
        Returns:
            Dictionary of capabilities
        """
        return {
            'psm_modes': {
                0: 'Orientation and script detection (OSD) only',
                1: 'Automatic page segmentation with OSD',
                2: 'Automatic page segmentation, but no OSD, or OCR',
                3: 'Fully automatic page segmentation, but no OSD (default)',
                4: 'Assume a single column of text of variable sizes',
                5: 'Assume a single uniform block of vertically aligned text',
                6: 'Assume a single uniform block of text',
                7: 'Treat the image as a single text line',
                8: 'Treat the image as a single word',
                9: 'Treat the image as a single word in a circle',
                10: 'Treat the image as a single character',
                11: 'Sparse text. Find as much text as possible in no particular order',
                12: 'Sparse text with OSD',
                13: 'Raw line. Treat the image as a single text line'
            },
            'oem_modes': {
                0: 'Legacy engine only',
                1: 'Neural nets LSTM engine only',
                2: 'Legacy + LSTM engines',
                3: 'Default, based on what is available'
            }
        }