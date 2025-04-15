"""
Main recognition engine for Bangla handwritten text.

This module coordinates between different OCR engines and preprocessing methods
to extract Bangla text from handwritten images.
"""

import os
import logging
import time
import subprocess
import tempfile
from typing import Dict, Union, Optional, List, Any, Tuple
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytesseract

from .preprocessing import (
    preprocess_image, 
    preprocess_all_methods, 
    create_visualization,
    check_image,
    PreprocessingResult
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BanglaRecognizer:
    """
    Main engine for recognizing Bangla handwritten text from images.
    
    This class provides methods to:
    1. Preprocess images for optimal OCR
    2. Recognize text using multiple OCR engines
    3. Combine and select the best results
    """
    
    def __init__(
        self, 
        use_tesseract: bool = True, 
        use_easyocr: bool = True,
        tesseract_path: Optional[str] = None,
        use_gpu: bool = False
    ):
        """
        Initialize the Bangla handwriting recognition engine.
        
        Args:
            use_tesseract: Whether to use Tesseract OCR
            use_easyocr: Whether to use EasyOCR
            tesseract_path: Path to Tesseract executable
            use_gpu: Whether to use GPU for EasyOCR (if available)
        """
        self.use_tesseract = use_tesseract
        self.use_easyocr = use_easyocr
        
        # Initialize Tesseract if requested
        if use_tesseract:
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                logger.info(f"Using Tesseract at: {tesseract_path}")
            
            # Check Tesseract availability
            try:
                langs = self.get_tesseract_languages()
                if langs:
                    logger.info(f"Available Tesseract languages: {langs}")
                else:
                    logger.warning("No languages found in Tesseract. Bengali recognition may not work.")
            except Exception as e:
                logger.error(f"Error initializing Tesseract: {e}")
                logger.warning("Tesseract unavailable. Will skip Tesseract processing.")
                self.use_tesseract = False
        
        # Initialize EasyOCR if requested
        self.easyocr_reader = None
        if use_easyocr:
            try:
                import easyocr
                logger.info("Initializing EasyOCR (this may take a moment)...")
                self.easyocr_reader = easyocr.Reader(['bn'], gpu=use_gpu)
                logger.info("EasyOCR initialized successfully")
            except ImportError:
                logger.error("EasyOCR not installed. Install with: pip install easyocr")
                logger.warning("EasyOCR unavailable. Will skip EasyOCR processing.")
                self.use_easyocr = False
            except Exception as e:
                logger.error(f"Error initializing EasyOCR: {e}")
                logger.warning("EasyOCR initialization failed. Will skip EasyOCR processing.")
                self.use_easyocr = False
    
    def is_tesseract_available(self) -> bool:
        """
        Check if Tesseract is available and working.
        
        Returns:
            Boolean indicating if Tesseract is available
        """
        if not self.use_tesseract:
            return False
            
        try:
            version = pytesseract.get_tesseract_version()
            return version is not None
        except Exception:
            return False
    
    def is_easyocr_available(self) -> bool:
        """
        Check if EasyOCR is available and initialized.
        
        Returns:
            Boolean indicating if EasyOCR is available
        """
        return self.use_easyocr and self.easyocr_reader is not None
    
    def get_tesseract_languages(self) -> List[str]:
        """
        Get list of available languages in Tesseract.
        
        Returns:
            List of language codes
        """
        if not self.use_tesseract:
            return []
            
        try:
            result = subprocess.run(
                [pytesseract.pytesseract.tesseract_cmd, '--list-langs'],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                logger.warning(f"Error getting Tesseract languages: {result.stderr}")
                return []
                
            # Parse languages from output
            langs = result.stdout.strip().split('\n')
            if langs and langs[0].startswith('List of'):
                langs = langs[1:]
                
            return langs
        except Exception as e:
            logger.error(f"Error getting Tesseract languages: {e}")
            return []
    
    def recognize_with_tesseract(
        self, 
        image: np.ndarray, 
        lang: str = 'ben'
    ) -> Tuple[str, float]:
        """
        Recognize text using Tesseract OCR.
        
        Args:
            image: Preprocessed image as numpy array
            lang: Language to use for recognition
            
        Returns:
            Tuple of (recognized text, confidence score)
        """
        if not self.use_tesseract:
            return "", 0.0
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Get available languages
        available_langs = self.get_tesseract_languages()
        
        # Ensure language is available
        if lang not in available_langs:
            logger.warning(f"Language '{lang}' not available in Tesseract. Using default.")
            lang = None
        
        # Tesseract configuration
        configs = [
            '--psm 6 --oem 3',  # Assume single block of text, LSTM only
            '--psm 4 --oem 3',  # Assume single column, LSTM only
            '--psm 3 --oem 3',  # Auto page segmentation, LSTM only
            '--psm 11 --oem 3', # Sparse text, LSTM only
            '--psm 13 --oem 3'  # Raw line, LSTM only
        ]
        
        best_text = ""
        best_score = 0.0
        
        for config in configs:
            try:
                # Build command with language if available
                cmd = f"{config}"
                if lang:
                    cmd += f" -l {lang}"
                
                # Get both text and data for confidence score
                text = pytesseract.image_to_string(pil_image, config=cmd)
                data = pytesseract.image_to_data(pil_image, config=cmd, output_type=pytesseract.Output.DICT)
                
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
                logger.warning(f"Error with Tesseract config {config}: {e}")
        
        # Post-process recognized text
        best_text = self._postprocess_text(best_text)
        
        return best_text, best_score
    
    def recognize_with_easyocr(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Recognize text using EasyOCR.
        
        Args:
            image: Image as numpy array (original, not preprocessed)
            
        Returns:
            Tuple of (recognized text, confidence score)
        """
        if not self.use_easyocr or self.easyocr_reader is None:
            return "", 0.0
        
        try:
            # EasyOCR works best with original image
            result = self.easyocr_reader.readtext(image)
            
            # Extract text and calculate average confidence
            texts = []
            confidences = []
            
            for bbox, text, conf in result:
                texts.append(text)
                confidences.append(conf)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Join all text blocks
            full_text = ' '.join(texts)
            
            # Post-process text
            full_text = self._postprocess_text(full_text)
            
            return full_text, avg_confidence
            
        except Exception as e:
            logger.error(f"Error with EasyOCR: {e}")
            return "", 0.0
    
    def _postprocess_text(self, text: str) -> str:
        """
        Post-process recognized text to improve quality.
        
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
    
    def recognize(
        self,
        image: Union[str, np.ndarray, Path],
        preprocess_method: str = 'auto',
        use_tesseract: Optional[bool] = None,
        use_easyocr: Optional[bool] = None,
        visualize: bool = False
    ) -> Dict[str, Any]:
        """
        Recognize Bangla text from an image.
        
        Args:
            image: Input image (path or numpy array)
            preprocess_method: Method for preprocessing
            use_tesseract: Override default Tesseract usage
            use_easyocr: Override default EasyOCR usage
            visualize: Whether to generate visualization
            
        Returns:
            Dictionary with recognition results
        """
        # Use instance defaults if not overridden
        use_tesseract = self.use_tesseract if use_tesseract is None else use_tesseract
        use_easyocr = self.use_easyocr if use_easyocr is None else use_easyocr
        
        # Load image
        img = check_image(image)
        original_img = img.copy()
        
        # Output dictionary
        result = {}
        
        # Process with multiple preprocessing methods if requested
        if preprocess_method == 'all':
            logger.info("Using all preprocessing methods")
            
            best_tesseract_text = ""
            best_tesseract_conf = 0
            best_tesseract_method = ""
            
            preprocess_results = preprocess_all_methods(img)
            
            # Generate visualization if requested
            if visualize:
                vis_path = os.path.join(tempfile.gettempdir(), f"recognize_viz_{int(time.time())}.png")
                create_visualization(preprocess_results, vis_path)
                result['visualization_path'] = vis_path
            
            # Tesseract OCR with multiple preprocessing methods
            if use_tesseract:
                logger.info("Running Tesseract with multiple preprocessing methods")
                
                for method, prep_result in preprocess_results.items():
                    try:
                        text, conf = self.recognize_with_tesseract(prep_result.processed)
                        
                        # Keep best result
                        if conf > best_tesseract_conf or (
                            conf == best_tesseract_conf and len(text) > len(best_tesseract_text)):
                            best_tesseract_text = text
                            best_tesseract_conf = conf
                            best_tesseract_method = method
                    except Exception as e:
                        logger.error(f"Error with Tesseract on {method} preprocessing: {e}")
                
                result['tesseract'] = {
                    'text': best_tesseract_text,
                    'confidence': best_tesseract_conf,
                    'method': best_tesseract_method
                }
            
            # EasyOCR (uses original image)
            if use_easyocr:
                logger.info("Running EasyOCR recognition")
                
                try:
                    text, conf = self.recognize_with_easyocr(original_img)
                    result['easyocr'] = {
                        'text': text,
                        'confidence': conf
                    }
                except Exception as e:
                    logger.error(f"Error with EasyOCR: {e}")
                    result['easyocr'] = {
                        'text': "",
                        'confidence': 0,
                        'error': str(e)
                    }
            
        else:
            # Process with a single preprocessing method
            logger.info(f"Using {preprocess_method} preprocessing method")
            
            # Preprocess the image
            prep_result = preprocess_image(img, method=preprocess_method, return_steps=True)
            
            # Generate visualization if requested
            if visualize:
                vis_path = os.path.join(tempfile.gettempdir(), f"recognize_viz_{int(time.time())}.png")
                create_visualization(prep_result, vis_path)
                result['visualization_path'] = vis_path
            
            # Tesseract OCR
            if use_tesseract:
                logger.info("Running Tesseract OCR")
                
                try:
                    text, conf = self.recognize_with_tesseract(prep_result.processed)
                    result['tesseract'] = {
                        'text': text,
                        'confidence': conf,
                        'method': preprocess_method
                    }
                except Exception as e:
                    logger.error(f"Error with Tesseract: {e}")
                    result['tesseract'] = {
                        'text': "",
                        'confidence': 0,
                        'error': str(e)
                    }
            
            # EasyOCR
            if use_easyocr:
                logger.info("Running EasyOCR recognition")
                
                try:
                    text, conf = self.recognize_with_easyocr(original_img)
                    result['easyocr'] = {
                        'text': text,
                        'confidence': conf
                    }
                except Exception as e:
                    logger.error(f"Error with EasyOCR: {e}")
                    result['easyocr'] = {
                        'text': "",
                        'confidence': 0,
                        'error': str(e)
                    }
        
        return result
