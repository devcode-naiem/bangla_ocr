"""
Visualization utilities for Bangla Handwriting Recognition.

This module provides functions for:
- Visualizing OCR results
- Creating comparison visualizations
- Generating annotated images with recognized text
"""

import os
import logging
import tempfile
import math
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from .image_utils import load_image, save_image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_preprocessing(
    original: np.ndarray,
    processed: Union[np.ndarray, Dict[str, np.ndarray]],
    output_path: Optional[str] = None,
    title: str = "Preprocessing Visualization",
    figsize: Tuple[int, int] = (12, 8)
) -> Optional[str]:
    """
    Create visualization of original and processed images.
    
    Args:
        original: Original image
        processed: Processed image or dictionary of named processed images
        output_path: Path to save visualization (if None, displayed inline)
        title: Title for the visualization
        figsize: Figure size (width, height) in inches
        
    Returns:
        Path to saved visualization if output_path is provided, None otherwise
    """
    # Determine number of processed images
    if isinstance(processed, dict):
        n_processed = len(processed)
        processed_images = list(processed.items())
    else:
        n_processed = 1
        processed_images = [("Processed", processed)]
    
    # Calculate layout
    if n_processed <= 3:
        # Single row layout for few images
        n_rows = 1
        n_cols = n_processed + 1  # +1 for original
    else:
        # Multi-row layout for many images
        n_rows = math.ceil((n_processed + 1) / 4)  # +1 for original
        n_cols = min(4, n_processed + 1)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Convert to array of axes for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = np.array(axes).reshape(-1)
    
    # Display original image
    if len(original.shape) == 3:
        # Color image - convert BGR to RGB for display
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        # Grayscale image
        axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Display processed images
    for i, (name, img) in enumerate(processed_images, 1):
        if i < len(axes):
            if len(img.shape) == 3:
                # Color image - convert BGR to RGB for display
                axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                # Grayscale image
                axes[i].imshow(img, cmap='gray')
            axes[i].set_title(name)
            axes[i].axis('off')
    
    # Hide any unused subplots
    for i in range(n_processed + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        plt.show()
        plt.close(fig)
        return None

def visualize_text_regions(
    image: np.ndarray,
    regions: List[Tuple[Tuple[int, int, int, int], str]],
    output_path: Optional[str] = None,
    colors: Union[Tuple[int, int, int], List[Tuple[int, int, int]]] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.8,
    with_text: bool = True,
    figsize: Tuple[int, int] = (12, 12)
) -> Optional[str]:
    """
    Visualize detected text regions on an image.
    
    Args:
        image: Original image
        regions: List of ((x, y, w, h), text) tuples
        output_path: Path to save visualization (if None, displayed inline)
        colors: Color or list of colors for boxes (BGR format)
        thickness: Line thickness for boxes
        font_scale: Scale for text font
        with_text: Whether to show the recognized text
        figsize: Figure size (width, height) in inches
        
    Returns:
        Path to saved visualization if output_path is provided, None otherwise
    """
    # Create a copy of the image
    vis_image = image.copy()
    
    # Use a single color if provided
    if not isinstance(colors, list):
        colors = [colors] * len(regions)
    
    # Ensure enough colors
    if len(colors) < len(regions):
        colors = colors * (len(regions) // len(colors) + 1)
    
    # Draw regions
    for i, ((x, y, w, h), text) in enumerate(regions):
        color = colors[i % len(colors)]
        
        # Draw rectangle
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, thickness)
        
        # Draw text if requested
        if with_text and text:
            # Calculate text position
            text_x = x
            text_y = y - 10 if y - 10 > 10 else y + h + 20
            
            # Draw background rectangle for text
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.rectangle(
                vis_image,
                (text_x, text_y - text_h - 5),
                (text_x + text_w, text_y + 5),
                (0, 0, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                vis_image, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
            )
    
    # Create figure for display
    plt.figure(figsize=figsize)
    
    # Display the image
    if len(vis_image.shape) == 3:
        # Color image - convert BGR to RGB for display
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    else:
        # Grayscale image
        plt.imshow(vis_image, cmap='gray')
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        plt.close()
        return None

def create_result_comparison(
    image: np.ndarray,
    results: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "OCR Result Comparison",
    figsize: Tuple[int, int] = (12, len(results) * 4 + 4)
) -> Optional[str]:
    """
    Create a comparison visualization of different OCR results.
    
    Args:
        image: Original image
        results: Dictionary with OCR results from different engines
                Format: {'engine_name': {'text': '...', 'confidence': 0.9, ...}, ...}
        output_path: Path to save visualization (if None, displayed inline)
        title: Title for the visualization
        figsize: Figure size (width, height) in inches
        
    Returns:
        Path to saved visualization if output_path is provided, None otherwise
    """
    # Determine number of engines
    n_engines = len(results)
    
    # Create figure
    fig, axes = plt.subplots(n_engines + 1, 1, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Display original image
    if len(image.shape) == 3:
        # Color image - convert BGR to RGB for display
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        # Grayscale image
        axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Display results for each engine
    for i, (engine_name, result) in enumerate(results.items(), 1):
        # Extract text and confidence
        text = result.get('text', '')
        confidence = result.get('confidence', 0)
        
        # Create a text display area
        axes[i].text(
            0.05, 0.5, 
            f"{engine_name} (Confidence: {confidence:.2f})\n\n{text}",
            verticalalignment='center',
            fontsize=12,
            fontfamily='monospace',
            wrap=True
        )
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        plt.show()
        plt.close(fig)
        return None

def create_progress_visualization(
    images: List[np.ndarray],
    titles: List[str],
    output_path: Optional[str] = None,
    main_title: str = "Processing Pipeline",
    figsize: Tuple[int, int] = None,
    with_arrows: bool = True
) -> Optional[str]:
    """
    Create a visualization of the processing pipeline.
    
    Args:
        images: List of images in processing order
        titles: List of titles for each image
        output_path: Path to save visualization (if None, displayed inline)
        main_title: Main title for the visualization
        figsize: Figure size (width, height) in inches
        with_arrows: Whether to show arrows between processing steps
        
    Returns:
        Path to saved visualization if output_path is provided, None otherwise
    """
    n_images = len(images)
    
    # Determine figure size if not provided
    if figsize is None:
        figsize = (n_images * 4, 6)
    
    # Create figure
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    fig.suptitle(main_title, fontsize=16)
    
    # Handle single image case
    if n_images == 1:
        axes = [axes]
    
    # Display images
    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 3:
            # Color image - convert BGR to RGB for display
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            # Grayscale image
            axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
        
        # Add arrow if not the last image and arrows are requested
        if with_arrows and i < n_images - 1:
            axes[i].annotate(
                '', 
                xy=(1.1, 0.5), xycoords=axes[i].transAxes,
                xytext=(1.5, 0.5), textcoords=axes[i].transAxes,
                arrowprops=dict(arrowstyle='->', color='black', lw=2)
            )
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        plt.show()
        plt.close(fig)
        return None

def create_annotated_image(
    image: np.ndarray,
    text: str,
    output_path: Optional[str] = None,
    font_size: int = 20,
    margin: int = 20,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    font_path: Optional[str] = None
) -> Optional[str]:
    """
    Create an image with annotation text below it.
    
    Args:
        image: Original image
        text: Text to append below the image
        output_path: Path to save the result (if None, a temp file is created)
        font_size: Size of the font for text
        margin: Margin around the text in pixels
        bg_color: Background color for the text area (RGB)
        text_color: Text color (RGB)
        font_path: Path to a TTF font file (if None, default is used)
        
    Returns:
        Path to the saved image
    """
    # Convert OpenCV BGR to PIL RGB if needed
    if len(image.shape) == 3:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_image = Image.fromarray(image)
    
    # Calculate text size
    try:
        # Try to use the specified font
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Use a default font
            try:
                # Try to use DejaVuSans which has good Unicode support
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except:
                # Fall back to default
                font = ImageFont.load_default()
        
        # Calculate text size
        draw = ImageDraw.Draw(pil_image)
        text_lines = text.split('\n')
        max_width = 0
        total_height = 0
        
        for line in text_lines:
            if line:
                line_bbox = draw.textbbox((0, 0), line, font=font)
                line_width = line_bbox[2] - line_bbox[0]
                line_height = line_bbox[3] - line_bbox[1]
                max_width = max(max_width, line_width)
                total_height += line_height + 5  # 5 pixels line spacing
        
        # Ensure minimum width
        max_width = max(max_width, pil_image.width)
        
        # Calculate text area
        text_area_height = total_height + 2 * margin
        
        # Create new image with space for text
        new_height = pil_image.height + text_area_height
        new_image = Image.new('RGB', (max_width, new_height), bg_color)
        
        # Paste original image
        new_image.paste(pil_image, ((max_width - pil_image.width) // 2, 0))
        
        # Draw text
        draw = ImageDraw.Draw(new_image)
        y_pos = pil_image.height + margin
        
        for line in text_lines:
            if line:
                line_bbox = draw.textbbox((0, 0), line, font=font)
                line_width = line_bbox[2] - line_bbox[0]
                line_height = line_bbox[3] - line_bbox[1]
                x_pos = (max_width - line_width) // 2
                draw.text((x_pos, y_pos), line, font=font, fill=text_color)
                y_pos += line_height + 5
        
    except Exception as e:
        logger.error(f"Error creating annotated image: {e}")
        return None
    
    # Save the image
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
    
    new_image.save(output_path)
    
    return output_path

def create_confidence_heatmap(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    confidences: List[float],
    output_path: Optional[str] = None,
    colormap: str = 'jet',
    alpha: float = 0.4,
    figsize: Tuple[int, int] = (12, 8)
) -> Optional[str]:
    """
    Create a heatmap visualization of text recognition confidence.
    
    Args:
        image: Original image
        boxes: List of (x, y, w, h) bounding boxes
        confidences: List of confidence values for each box
        output_path: Path to save visualization (if None, displayed inline)
        colormap: Matplotlib colormap to use
        alpha: Transparency of the heatmap overlay
        figsize: Figure size (width, height) in inches
        
    Returns:
        Path to saved visualization if output_path is provided, None otherwise
    """
    # Create a blank heatmap
    heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    
    # Fill the heatmap with confidence values
    for (x, y, w, h), conf in zip(boxes, confidences):
        heatmap[y:y+h, x:x+w] = conf
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Display the original image
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    
    # Overlay the heatmap
    plt.imshow(heatmap, cmap=colormap, alpha=alpha)
    
    # Add a colorbar
    cbar = plt.colorbar()
    cbar.set_label('Confidence')
    
    plt.title('Text Recognition Confidence Heatmap')
    plt.axis('off')
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        plt.close()
        return None

def create_html_report(
    image_path: str,
    results: Dict[str, Dict[str, Any]],
    output_path: str,
    title: str = "Bangla OCR Results",
    include_images: bool = True
) -> str:
    """
    Create an HTML report of OCR results.
    
    Args:
        image_path: Path to the original image
        results: Dictionary with OCR results from different engines
        output_path: Path to save the HTML report
        title: Title for the report
        include_images: Whether to include images in the report
        
    Returns:
        Path to the saved HTML report
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create HTML content
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .result {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        .image {{ margin-bottom: 20px; }}
        img {{ max-width: 100%; }}
        .engine {{ font-weight: bold; color: #3498db; }}
        .confidence {{ color: #27ae60; }}
        .text {{ white-space: pre-wrap; padding: 10px; background-color: #f9f9f9; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
"""
    
    # Add original image if requested
    if include_images:
        image_filename = os.path.basename(image_path)
        html += f"""
        <div class="image">
            <h2>Original Image</h2>
            <img src="{image_filename}" alt="Original Image">
        </div>
"""
    
    # Add results for each engine
    for engine_name, result in results.items():
        text = result.get('text', '')
        confidence = result.get('confidence', 0)
        
        html += f"""
        <div class="result">
            <h2 class="engine">{engine_name}</h2>
            <p class="confidence">Confidence: {confidence:.2f}</p>
            <div class="text">{text}</div>
        </div>
"""
    
    # Close HTML
    html += """
    </div>
</body>
</html>
"""
    
    # Write HTML to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    # If including images, copy the original image to the output directory
    if include_images:
        output_image_path = os.path.join(output_dir, os.path.basename(image_path))
        if not os.path.exists(output_image_path):
            try:
                from shutil import copyfile
                copyfile(image_path, output_image_path)
            except Exception as e:
                logger.error(f"Error copying image: {e}")
    
    return output_path