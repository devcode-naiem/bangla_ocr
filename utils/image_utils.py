"""
Image handling utilities for Bangla Handwriting Recognition.

This module provides functions for:
- Loading images from various sources
- Converting between image formats
- Basic image manipulation
- Image validation and preparation
"""

import os
import io
import base64
import logging
import tempfile
import requests
from typing import Union, Optional, Tuple, List, Dict, Any
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_url(path: str) -> bool:
    """
    Check if a string is a valid URL.
    
    Args:
        path: String to check
        
    Returns:
        Boolean indicating if the string is a URL
    """
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except:
        return False

def load_image(
    source: Union[str, Path, np.ndarray, bytes, Image.Image],
    return_format: str = 'np',
    grayscale: bool = False,
    resize: Optional[Tuple[int, int]] = None,
    max_size: Optional[int] = None
) -> Union[np.ndarray, Image.Image]:
    """
    Load an image from various sources.
    
    Args:
        source: Image source (file path, URL, numpy array, bytes, PIL Image)
        return_format: Return format ('np' for numpy array, 'pil' for PIL Image)
        grayscale: Whether to convert to grayscale
        resize: Optional (width, height) to resize the image to
        max_size: Optional maximum size for any dimension
        
    Returns:
        Loaded image in the specified format
        
    Raises:
        ValueError: If the image cannot be loaded or is invalid
    """
    img = None
    
    # Handle different input types
    if isinstance(source, (str, Path)):
        source_str = str(source)
        
        # Handle URL
        if is_url(source_str):
            try:
                response = requests.get(source_str, stream=True, timeout=10)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
            except Exception as e:
                raise ValueError(f"Failed to load image from URL: {e}")
        
        # Handle file path
        else:
            if not os.path.exists(source_str):
                raise ValueError(f"Image file not found: {source_str}")
            
            try:
                img = Image.open(source_str)
            except Exception as e:
                raise ValueError(f"Failed to load image from file: {e}")
    
    # Handle numpy array
    elif isinstance(source, np.ndarray):
        if len(source.shape) < 2:
            raise ValueError("Invalid image array: must have at least 2 dimensions")
        
        # Convert BGR to RGB if color image (OpenCV default is BGR)
        if len(source.shape) == 3 and source.shape[2] == 3:
            img_array = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        else:
            img_array = source
            
        img = Image.fromarray(img_array.astype('uint8'))
    
    # Handle bytes
    elif isinstance(source, bytes):
        try:
            img = Image.open(io.BytesIO(source))
        except Exception as e:
            raise ValueError(f"Failed to load image from bytes: {e}")
    
    # Handle PIL Image
    elif isinstance(source, Image.Image):
        img = source
    
    else:
        raise ValueError(f"Unsupported image source type: {type(source)}")
    
    # Ensure the image is loaded
    if img is None:
        raise ValueError("Failed to load image")
    
    # Convert to grayscale if requested
    if grayscale:
        img = img.convert('L')
    
    # Resize if requested
    if resize:
        img = img.resize(resize, Image.LANCZOS)
    
    # Limit size if requested
    if max_size and (img.width > max_size or img.height > max_size):
        ratio = max_size / max(img.width, img.height)
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Convert to the requested format
    if return_format.lower() == 'np':
        # Convert PIL to numpy
        img_array = np.array(img)
        
        # Convert RGB to BGR for OpenCV compatibility
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_array
    
    elif return_format.lower() == 'pil':
        return img
    
    else:
        raise ValueError(f"Unsupported return format: {return_format}")

def save_image(
    image: Union[np.ndarray, Image.Image],
    path: str,
    quality: int = 95
) -> str:
    """
    Save an image to a file.
    
    Args:
        image: Image to save (numpy array or PIL Image)
        path: Path where to save the image
        quality: JPEG quality (1-100) if saving as JPEG
        
    Returns:
        Path where the image was saved
        
    Raises:
        ValueError: If the image cannot be saved
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    try:
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if color image
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Determine format based on extension
        ext = os.path.splitext(path)[1].lower()
        
        # Save with appropriate parameters
        if ext in ['.jpg', '.jpeg']:
            pil_image.save(path, quality=quality, optimize=True)
        elif ext == '.png':
            pil_image.save(path, optimize=True)
        else:
            pil_image.save(path)
            
        return path
    
    except Exception as e:
        raise ValueError(f"Failed to save image: {e}")

def image_to_base64(
    image: Union[np.ndarray, Image.Image, str, Path],
    format: str = 'png'
) -> str:
    """
    Convert an image to a base64 string.
    
    Args:
        image: Image to convert (numpy array, PIL Image, or path)
        format: Output format (png, jpeg, etc.)
        
    Returns:
        Base64 encoded string
        
    Raises:
        ValueError: If the image cannot be converted
    """
    # Load image if path is provided
    if isinstance(image, (str, Path)):
        image = load_image(image, return_format='pil')
    
    # Convert numpy to PIL if needed
    if isinstance(image, np.ndarray):
        # Convert BGR to RGB if color image
        if len(image.shape) == 3 and image.shape[2] == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Convert PIL Image to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/{format};base64,{img_str}"

def base64_to_image(
    base64_str: str,
    return_format: str = 'np'
) -> Union[np.ndarray, Image.Image]:
    """
    Convert a base64 string to an image.
    
    Args:
        base64_str: Base64 encoded image string
        return_format: Return format ('np' for numpy array, 'pil' for PIL Image)
        
    Returns:
        Image in the specified format
        
    Raises:
        ValueError: If the string cannot be decoded
    """
    try:
        # Remove data URL prefix if present
        if base64_str.startswith('data:image'):
            base64_str = base64_str.split(',', 1)[1]
        
        # Decode base64
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        
        # Return in the requested format
        if return_format.lower() == 'pil':
            return img
        else:
            img_array = np.array(img)
            # Convert RGB to BGR for OpenCV compatibility
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return img_array
            
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}")

def download_image(
    url: str,
    output_path: Optional[str] = None,
    timeout: int = 10
) -> str:
    """
    Download an image from a URL.
    
    Args:
        url: URL of the image
        output_path: Path where to save the image (if None, a temp file is created)
        timeout: Connection timeout in seconds
        
    Returns:
        Path to the downloaded image
        
    Raises:
        ValueError: If the image cannot be downloaded
    """
    try:
        # Download the image
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        # Create temporary file if no output path
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix='.jpg')
            os.close(fd)
        
        # Save the image
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return output_path
        
    except Exception as e:
        raise ValueError(f"Failed to download image from {url}: {e}")

def extract_text_regions(
    image: np.ndarray,
    min_area: int = 100,
    margin: int = 10
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Extract regions containing text from an image.
    
    Args:
        image: Input image
        min_area: Minimum area for a region to be considered
        margin: Margin to add around regions
        
    Returns:
        List of (region_image, (x, y, w, h)) tuples
    """
    # Ensure image is grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract regions
    regions = []
    for contour in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by area
        if w * h < min_area:
            continue
        
        # Add margin
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        # Extract region
        if len(image.shape) == 3:
            region = image[y:y+h, x:x+w].copy()
        else:
            region = image[y:y+h, x:x+w].copy()
        
        regions.append((region, (x, y, w, h)))
    
    return regions

def correct_skew(
    image: np.ndarray,
    delta: float = 1.0,
    limit: float = 5.0
) -> np.ndarray:
    """
    Correct skew in an image.
    
    Args:
        image: Input image
        delta: Delta angle for searching
        limit: Maximum angle to consider
        
    Returns:
        Deskewed image
    """
    # Ensure image is grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Apply dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)
    
    # Find contours
    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Find largest contour
    for c in contours:
        rect = cv2.minAreaRect(c)
        angle = rect[2]
        
        # Adjust angle
        if angle < -45:
            angle = 90 + angle
        
        # Limit angle
        if abs(angle) > limit:
            angle = 0
        
        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    # If no contours found, return original
    return image

def get_image_info(image: Union[np.ndarray, Image.Image, str, Path]) -> Dict[str, Any]:
    """
    Get information about an image.
    
    Args:
        image: Image to analyze
        
    Returns:
        Dictionary with image information
    """
    # Load image if path is provided
    if isinstance(image, (str, Path)):
        path = str(image)
        pil_image = load_image(image, return_format='pil')
        file_info = {
            'path': path,
            'filename': os.path.basename(path),
            'size_bytes': os.path.getsize(path) if os.path.exists(path) else None,
        }
    else:
        pil_image = load_image(image, return_format='pil')
        file_info = {
            'path': None,
            'filename': None,
            'size_bytes': None,
        }
    
    # Get image properties
    width, height = pil_image.size
    mode = pil_image.mode
    format = pil_image.format
    
    # Calculate histogram for non-palette images
    histogram = None
    if mode not in ['P']:
        try:
            hist = pil_image.histogram()
            if mode == 'L':
                # Grayscale histogram
                histogram = {
                    'gray': hist
                }
            elif mode in ['RGB', 'RGBA']:
                # RGB histogram
                histogram = {
                    'r': hist[0:256],
                    'g': hist[256:512],
                    'b': hist[512:768]
                }
        except:
            pass
    
    # Calculate basic statistics for grayscale images
    stats = None
    if mode == 'L':
        img_array = np.array(pil_image)
        stats = {
            'min': int(np.min(img_array)),
            'max': int(np.max(img_array)),
            'mean': float(np.mean(img_array)),
            'std': float(np.std(img_array)),
        }
    
    return {
        **file_info,
        'width': width,
        'height': height,
        'mode': mode,
        'format': format,
        'histogram': histogram,
        'stats': stats
    }

def enhance_image(
    image: np.ndarray,
    operation: str = 'auto',
    **kwargs
) -> np.ndarray:
    """
    Apply various enhancements to an image.
    
    Args:
        image: Input image
        operation: Enhancement operation to apply ('auto', 'contrast', 'sharpen', 'denoise')
        **kwargs: Additional parameters for the enhancement operation
        
    Returns:
        Enhanced image
    """
    # Ensure we have a copy to work with
    enhanced = image.copy()
    
    if operation == 'auto':
        # Analyze image to determine best enhancement
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate image statistics
        mean = np.mean(gray)
        std = np.std(gray)
        
        # Choose enhancement based on image characteristics
        if std < 30:  # Low contrast
            logger.info("Detected low contrast, applying contrast enhancement")
            enhanced = enhance_image(image, 'contrast', **kwargs)
        elif std > 60:  # Noisy image
            logger.info("Detected high noise, applying denoising")
            enhanced = enhance_image(image, 'denoise', **kwargs)
        else:
            logger.info("Applying sharpening")
            enhanced = enhance_image(image, 'sharpen', **kwargs)
    
    elif operation == 'contrast':
        # Get parameters
        clip_limit = kwargs.get('clip_limit', 2.0)
        tile_grid_size = kwargs.get('tile_grid_size', (8, 8))
        
        # Convert to LAB color space (if color image)
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            cl = clahe.apply(l)
            
            # Merge channels
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            # Apply CLAHE directly to grayscale
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced = clahe.apply(image)
    
    elif operation == 'sharpen':
        # Get parameters
        kernel_size = kwargs.get('kernel_size', 5)
        sigma = kwargs.get('sigma', 1.0)
        amount = kwargs.get('amount', 1.5)
        threshold = kwargs.get('threshold', 0)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        # Calculate unsharp mask
        sharpened = cv2.addWeighted(image, amount, blurred, 1.0 - amount, 0)
        
        # Apply threshold if requested
        if threshold > 0:
            low_contrast_mask = abs(image - blurred) < threshold
            enhanced = image.copy()
            enhanced[~low_contrast_mask] = sharpened[~low_contrast_mask]
        else:
            enhanced = sharpened
    
    elif operation == 'denoise':
        # Get parameters
        h = kwargs.get('h', 10)
        template_size = kwargs.get('template_size', 7)
        search_size = kwargs.get('search_size', 21)
        
        # Apply denoising
        if len(image.shape) == 3:
            enhanced = cv2.fastNlMeansDenoisingColored(
                image, None, h, h, template_size, search_size
            )
        else:
            enhanced = cv2.fastNlMeansDenoising(
                image, None, h, template_size, search_size
            )
    
    else:
        logger.warning(f"Unknown enhancement operation: {operation}")
    
    return enhanced