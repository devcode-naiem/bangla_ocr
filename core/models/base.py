"""
Abstract base class for OCR models used in Bangla Handwriting Recognition.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path


class OCRModel(ABC):
    """
    Abstract base class for OCR models.
    
    This defines the interface that all OCR models must implement.
    """
    
    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the OCR model.
        
        Args:
            **kwargs: Model-specific initialization parameters
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the model is available and properly initialized.
        
        Returns:
            Boolean indicating if the model is available
        """
        pass
    
    @abstractmethod
    def get_available_languages(self) -> List[str]:
        """
        Get a list of available languages supported by this model.
        
        Returns:
            List of language codes
        """
        pass
    
    @abstractmethod
    def recognize(self, image: np.ndarray, **kwargs) -> Tuple[str, float]:
        """
        Recognize text from an image.
        
        Args:
            image: Input image as numpy array
            **kwargs: Additional parameters for recognition
            
        Returns:
            Tuple of (recognized text, confidence score)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this OCR model.
        
        Returns:
            Model name as string
        """
        pass
    
    @abstractmethod
    def get_version(self) -> Optional[str]:
        """
        Get the version of this OCR model.
        
        Returns:
            Version string or None if not available
        """
        pass
    
    @property
    def name(self) -> str:
        """
        Property to get model name.
        
        Returns:
            Model name
        """
        return self.get_name()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about this model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'name': self.get_name(),
            'version': self.get_version(),
            'available': self.is_available(),
            'languages': self.get_available_languages()
        }
    
    def __str__(self) -> str:
        """
        String representation of the model.
        
        Returns:
            String describing the model
        """
        info = self.get_model_info()
        return f"{info['name']} (v{info['version']}) - Available: {info['available']}"