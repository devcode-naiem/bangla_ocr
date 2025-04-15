"""
Image preprocessing techniques for improving OCR accuracy on Bangla handwriting.

This module provides various preprocessing methods that can be applied to
handwritten Bangla text images before OCR processing.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple, Union, List, Optional, Callable
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingResult:
    """Container for preprocessing result data"""
    original: np.ndarray
    processed: np.ndarray
    method_name: str
    params: Dict = None
    intermediate_steps: Dict[str, np.ndarray] = None


def check_image(image: Union[str, np.ndarray, Path]) -> np.ndarray:
    """
    Check and load image from various input types.
    
    Args:
        image: Input image as path string, Path object, or numpy array
        
    Returns:
        Loaded image as numpy array
        
    Raises:
        ValueError: If image cannot be loaded or is invalid
    """
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"Could not read image at {image}")
        return img
    elif isinstance(image, np.ndarray):
        if len(image.shape) < 2:
            raise ValueError("Invalid image array: must have at least 2 dimensions")
        return image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def resize_if_needed(image: np.ndarray, max_size: int = 1920) -> np.ndarray:
    """
    Resize image if either dimension exceeds max_size while preserving aspect ratio.
    
    Args:
        image: Input image
        max_size: Maximum allowed dimension size
        
    Returns:
        Resized image or original if no resize needed
    """
    h, w = image.shape[:2]
    
    # Skip if image is already smaller than max_size
    if max(h, w) <= max_size:
        return image
    
    # Calculate new dimensions
    if h > w:
        new_h = max_size
        new_w = int(w * (max_size / h))
    else:
        new_w = max_size
        new_h = int(h * (max_size / w))
    
    # Resize the image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.info(f"Resized image from {w}x{h} to {new_w}x{new_h}")
    
    return resized


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale if it's color.
    
    Args:
        image: Input image
        
    Returns:
        Grayscale image
    """
    # Convert to grayscale if the image has 3 channels
    if len(image.shape) == 3 and image.shape[2] > 1:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def denoise_image(image: np.ndarray, method: str = 'gaussian', **kwargs) -> np.ndarray:
    """
    Apply noise reduction to image.
    
    Args:
        image: Input grayscale image
        method: Denoising method ('gaussian', 'bilateral', 'nlmeans', 'median')
        **kwargs: Additional parameters for the denoising method
        
    Returns:
        Denoised image
    """
    if method == 'gaussian':
        ksize = kwargs.get('ksize', 5)
        sigma = kwargs.get('sigma', 0)
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)
    
    elif method == 'bilateral':
        d = kwargs.get('d', 9)
        sigma_color = kwargs.get('sigma_color', 75)
        sigma_space = kwargs.get('sigma_space', 75)
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    elif method == 'nlmeans':
        h = kwargs.get('h', 10)
        template_window_size = kwargs.get('template_window_size', 7)
        search_window_size = kwargs.get('search_window_size', 21)
        return cv2.fastNlMeansDenoising(image, None, h, template_window_size, search_window_size)
    
    elif method == 'median':
        ksize = kwargs.get('ksize', 5)
        return cv2.medianBlur(image, ksize)
    
    else:
        logger.warning(f"Unknown denoising method '{method}', using gaussian blur")
        return cv2.GaussianBlur(image, (5, 5), 0)


def enhance_contrast(image: np.ndarray, method: str = 'clahe', **kwargs) -> np.ndarray:
    """
    Enhance the contrast of the image.
    
    Args:
        image: Input grayscale image
        method: Enhancement method ('clahe', 'histogram', 'stretch')
        **kwargs: Additional parameters
        
    Returns:
        Contrast-enhanced image
    """
    if method == 'clahe':
        clip_limit = kwargs.get('clip_limit', 2.0)
        tile_grid_size = kwargs.get('tile_grid_size', (8, 8))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    
    elif method == 'histogram':
        return cv2.equalizeHist(image)
    
    elif method == 'stretch':
        # Simple contrast stretching
        min_val = np.min(image)
        max_val = np.max(image)
        if min_val == max_val:
            return image
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    else:
        logger.warning(f"Unknown contrast enhancement method '{method}', using CLAHE")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)


def apply_threshold(image: np.ndarray, method: str = 'adaptive', **kwargs) -> np.ndarray:
    """
    Apply thresholding to convert grayscale image to binary.
    
    Args:
        image: Input grayscale image
        method: Thresholding method ('adaptive', 'otsu', 'local', 'triangle')
        **kwargs: Additional parameters
        
    Returns:
        Binary image
    """
    if method == 'adaptive':
        block_size = kwargs.get('block_size', 11)
        c = kwargs.get('c', 2)
        
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
            
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, block_size, c
        )
    
    elif method == 'otsu':
        # Otsu's thresholding
        _, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary
    
    elif method == 'local':
        # Custom local thresholding with larger window
        block_size = kwargs.get('block_size', 25)
        c = kwargs.get('c', 15)
        
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
            
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY_INV, block_size, c
        )
    
    elif method == 'triangle':
        # Triangle thresholding - good for bimodal images
        _, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE
        )
        return binary
    
    else:
        logger.warning(f"Unknown thresholding method '{method}', using Otsu's method")
        _, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary


def apply_morphology(image: np.ndarray, operations: List[Tuple[str, Dict]] = None) -> np.ndarray:
    """
    Apply a sequence of morphological operations to the binary image.
    
    Args:
        image: Input binary image
        operations: List of (operation_name, params) tuples
                   Available operations: 'erode', 'dilate', 'open', 'close', 'gradient'
        
    Returns:
        Processed image
    """
    if operations is None:
        # Default: open to remove noise, then close to connect components
        operations = [
            ('open', {'kernel_size': (2, 2), 'iterations': 1}),
            ('close', {'kernel_size': (1, 3), 'iterations': 1})
        ]
    
    result = image.copy()
    
    for op_name, params in operations:
        # Get kernel parameters
        kernel_size = params.get('kernel_size', (3, 3))
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
            
        iterations = params.get('iterations', 1)
        
        # Create kernel
        kernel = np.ones(kernel_size, np.uint8)
        
        # Apply morphological operation
        if op_name == 'erode':
            result = cv2.erode(result, kernel, iterations=iterations)
        elif op_name == 'dilate':
            result = cv2.dilate(result, kernel, iterations=iterations)
        elif op_name == 'open':
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif op_name == 'close':
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        elif op_name == 'gradient':
            result = cv2.morphologyEx(result, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
        else:
            logger.warning(f"Unknown morphological operation '{op_name}', skipping")
    
    return result


def preprocess_adaptive(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Apply adaptive thresholding preprocessing pipeline.
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (processed image, intermediate steps)
    """
    steps = {}
    
    # Normalize to grayscale
    gray = normalize_image(image)
    steps['gray'] = gray
    
    # Apply bilateral filter to preserve edges
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    steps['bilateral'] = bilateral
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    steps['binary'] = binary
    
    # Morphological operations
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    steps['opening'] = opening
    
    kernel = np.ones((1, 3), np.uint8)  # Horizontal kernel to connect chars
    result = cv2.dilate(opening, kernel, iterations=1)
    steps['dilated'] = result
    
    return result, steps


def preprocess_otsu(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Apply Otsu thresholding preprocessing pipeline.
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (processed image, intermediate steps)
    """
    steps = {}
    
    # Normalize to grayscale
    gray = normalize_image(image)
    steps['gray'] = gray
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    steps['blurred'] = blurred
    
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    steps['binary'] = binary
    
    # Morphological operations
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    steps['opening'] = opening
    
    kernel = np.ones((1, 2), np.uint8)  # Horizontal kernel
    result = cv2.dilate(opening, kernel, iterations=1)
    steps['dilated'] = result
    
    return result, steps


def preprocess_local(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Apply local thresholding with contrast enhancement.
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (processed image, intermediate steps)
    """
    steps = {}
    
    # Normalize to grayscale
    gray = normalize_image(image)
    steps['gray'] = gray
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    steps['blurred'] = blurred
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    steps['enhanced'] = enhanced
    
    # Apply adaptive threshold with larger block size
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 19, 9
    )
    steps['binary'] = binary
    
    # Morphological operations
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    steps['opening'] = opening
    
    kernel = np.ones((1, 3), np.uint8)  # Wider horizontal kernel
    result = cv2.dilate(opening, kernel, iterations=1)
    steps['dilated'] = result
    
    return result, steps


def preprocess_denoise(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Apply denoising-focused preprocessing pipeline.
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (processed image, intermediate steps)
    """
    steps = {}
    
    # Normalize to grayscale
    gray = normalize_image(image)
    steps['gray'] = gray
    
    # Apply non-local means denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    steps['denoised'] = denoised
    
    # Enhance edges
    edge_enhanced = cv2.Laplacian(denoised, cv2.CV_8U, ksize=3)
    edge_enhanced = cv2.convertScaleAbs(edge_enhanced)
    steps['edges'] = edge_enhanced
    
    # Combine with original
    sharpened = cv2.addWeighted(denoised, 1.5, edge_enhanced, -0.5, 0)
    steps['sharpened'] = sharpened
    
    # Apply threshold
    _, result = cv2.threshold(
        sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    steps['binary'] = result
    
    return result, steps


def preprocess_image(
    image: Union[str, np.ndarray, Path],
    method: str = 'auto',
    return_steps: bool = False
) -> Union[np.ndarray, PreprocessingResult]:
    """
    Apply preprocessing pipeline to an image.
    
    Args:
        image: Input image (file path or numpy array)
        method: Preprocessing method ('auto', 'adaptive', 'otsu', 'local', 'denoise')
        return_steps: Whether to return intermediate steps
        
    Returns:
        Preprocessed image or PreprocessingResult object if return_steps=True
    """
    # Load and check image
    img = check_image(image)
    
    # Resize if needed to improve performance
    img = resize_if_needed(img)
    
    # Original image for reference
    original = img.copy()
    
    # Dictionary to store intermediate steps
    steps = {'original': original}
    
    # Apply the selected preprocessing method
    if method == 'auto':
        # Auto method selection based on image characteristics
        
        # Convert to grayscale
        gray = normalize_image(img)
        
        # Calculate image statistics
        mean = np.mean(gray)
        std = np.std(gray)
        
        logger.info(f"Image statistics: mean={mean:.2f}, std={std:.2f}")
        
        # Decision logic
        if std < 40:  # Low contrast image
            logger.info("Detected low contrast image, using local enhancement")
            result, method_steps = preprocess_local(img)
            method = 'local'
        elif mean < 100:  # Dark image
            logger.info("Detected dark image, using Otsu thresholding")
            result, method_steps = preprocess_otsu(img)
            method = 'otsu'
        elif std > 60:  # Noisy image
            logger.info("Detected noisy image, using denoising pipeline")
            result, method_steps = preprocess_denoise(img)
            method = 'denoise'
        else:  # Default case
            logger.info("Using adaptive thresholding")
            result, method_steps = preprocess_adaptive(img)
            method = 'adaptive'
            
        steps.update(method_steps)
    
    elif method == 'adaptive':
        result, method_steps = preprocess_adaptive(img)
        steps.update(method_steps)
    
    elif method == 'otsu':
        result, method_steps = preprocess_otsu(img)
        steps.update(method_steps)
    
    elif method == 'local':
        result, method_steps = preprocess_local(img)
        steps.update(method_steps)
    
    elif method == 'denoise':
        result, method_steps = preprocess_denoise(img)
        steps.update(method_steps)
    
    elif method == 'all':
        # Just return original for 'all' method - the caller will handle multiple methods
        if return_steps:
            return PreprocessingResult(
                original=original,
                processed=original,
                method_name='all',
                params={},
                intermediate_steps={'original': original}
            )
        return original
    
    else:
        logger.warning(f"Unknown preprocessing method '{method}', using adaptive thresholding")
        result, method_steps = preprocess_adaptive(img)
        method = 'adaptive'
        steps.update(method_steps)
    
    # Return preprocessed image with or without steps
    if return_steps:
        return PreprocessingResult(
            original=original,
            processed=result,
            method_name=method,
            params={},
            intermediate_steps=steps
        )
    
    return result


def preprocess_all_methods(
    image: Union[str, np.ndarray, Path]
) -> Dict[str, PreprocessingResult]:
    """
    Apply all preprocessing methods to an image.
    
    Args:
        image: Input image (file path or numpy array)
        
    Returns:
        Dictionary of method name to PreprocessingResult
    """
    # Load and check image
    img = check_image(image)
    
    # Resize if needed to improve performance
    img = resize_if_needed(img)
    
    # Apply all preprocessing methods
    methods = ['adaptive', 'otsu', 'local', 'denoise']
    results = {}
    
    for method in methods:
        results[method] = preprocess_image(img, method=method, return_steps=True)
    
    return results


def create_visualization(
    results: Union[PreprocessingResult, Dict[str, PreprocessingResult]],
    output_path: Optional[str] = None
) -> Optional[str]:
    """
    Create visualization of preprocessing results.
    
    Args:
        results: Single PreprocessingResult or dictionary of results
        output_path: Path to save visualization image
        
    Returns:
        Path to saved visualization image if output_path is provided, None otherwise
    """
    if isinstance(results, PreprocessingResult):
        # Single result visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Original image
        if len(results.original.shape) == 3:
            axes[0].imshow(cv2.cvtColor(results.original, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(results.original, cmap='gray')
        
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Processed image
        axes[1].imshow(results.processed, cmap='gray')
        axes[1].set_title(f'Processed ({results.method_name})')
        axes[1].axis('off')
        
        plt.tight_layout()
        
    else:
        # Multiple results visualization
        num_methods = len(results)
        fig, axes = plt.subplots(2, num_methods + 1, figsize=(4 * (num_methods + 1), 8))
        
        # Original image (first column)
        first_result = next(iter(results.values()))
        
        if len(first_result.original.shape) == 3:
            axes[0, 0].imshow(cv2.cvtColor(first_result.original, cv2.COLOR_BGR2RGB))
        else:
            axes[0, 0].imshow(first_result.original, cmap='gray')
        
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Empty cell below original
        axes[1, 0].axis('off')
        
        # Display each method's result
        for i, (method_name, result) in enumerate(results.items(), 1):
            # Binary result
            axes[0, i].imshow(result.processed, cmap='gray')
            axes[0, i].set_title(f'{method_name.capitalize()}')
            axes[0, i].axis('off')
            
            # If available, show an intermediate step (e.g., edges or enhanced)
            steps = result.intermediate_steps
            if 'enhanced' in steps:
                axes[1, i].imshow(steps['enhanced'], cmap='gray')
                axes[1, i].set_title(f'Enhanced')
            elif 'edges' in steps:
                axes[1, i].imshow(steps['edges'], cmap='gray')
                axes[1, i].set_title(f'Edges')
            elif 'binary' in steps:
                axes[1, i].imshow(steps['binary'], cmap='gray')
                axes[1, i].set_title(f'Binary')
            else:
                axes[1, i].axis('off')
        
        plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path
    
    # Show plot if no output path
    plt.show()
    plt.close(fig)
    return None