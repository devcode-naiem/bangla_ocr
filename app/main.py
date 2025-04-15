"""
Command-line application entry point for Bangla Handwriting Recognition
"""

import argparse
import sys
import os
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

from core.recognizer import BanglaRecognizer
from . import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_images(
    image_paths: List[str],
    engine: str = "both",
    preprocess: str = "auto",
    output_dir: Optional[str] = None,
    visualize: bool = False,
    save_text: bool = False,
    save_json: bool = False
) -> Dict[str, Any]:
    """
    Process a list of images for Bangla text recognition
    
    Args:
        image_paths: List of paths to images
        engine: OCR engine to use (tesseract, easyocr, or both)
        preprocess: Preprocessing method
        output_dir: Directory to save results
        visualize: Whether to generate visualization
        save_text: Whether to save recognized text to file
        save_json: Whether to save JSON output
        
    Returns:
        Dictionary with results for each image
    """
    # Validate engine parameter
    if engine not in ["tesseract", "easyocr", "both"]:
        logger.error("Invalid engine parameter. Must be 'tesseract', 'easyocr', or 'both'")
        sys.exit(1)
    
    # Validate preprocessing method
    if preprocess not in ["auto", "adaptive", "otsu", "local", "denoise", "all"]:
        logger.error("Invalid preprocess parameter")
        sys.exit(1)
    
    # Set up output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = config.RESULT_DIR
    
    # Initialize recognizer
    try:
        logger.info("Initializing recognizer...")
        recognizer = BanglaRecognizer(
            use_tesseract=engine in ["tesseract", "both"],
            use_easyocr=engine in ["easyocr", "both"],
            tesseract_path=config.TESSERACT_PATH,
            use_gpu=config.EASYOCR_GPU
        )
        logger.info("Recognizer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize recognizer: {e}")
        sys.exit(1)
    
    # Store all results
    all_results = {}
    
    # Process each image
    for image_path in image_paths:
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            all_results[image_path] = {"error": "File not found"}
            continue
        
        try:
            logger.info(f"Processing image: {image_path}")
            start_time = time.time()
            
            # Process the image
            result = recognizer.recognize(
                image_path,
                preprocess_method=preprocess,
                use_tesseract=engine in ["tesseract", "both"],
                use_easyocr=engine in ["easyocr", "both"],
                visualize=visualize
            )
            
            # Add processing time
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            
            # Store result for return
            all_results[image_path] = result
            
            # Determine best result for display
            if engine == "both":
                easyocr_confidence = result.get('easyocr', {}).get('confidence', 0)
                tesseract_confidence = result.get('tesseract', {}).get('confidence', 0)
                
                if easyocr_confidence > tesseract_confidence:
                    best_text = result.get('easyocr', {}).get('text', '')
                    best_engine = "easyocr"
                    confidence = easyocr_confidence
                else:
                    best_text = result.get('tesseract', {}).get('text', '')
                    best_engine = "tesseract"
                    confidence = tesseract_confidence
                
                # Print all results
                print(f"\n=== Results for {os.path.basename(image_path)} ===")
                print(f"Processing time: {processing_time:.2f} seconds")
                
                print("\n--- EasyOCR Result ---")
                print(result.get('easyocr', {}).get('text', 'No text detected'))
                print(f"Confidence: {easyocr_confidence:.2f}")
                
                print("\n--- Tesseract Result ---")
                print(result.get('tesseract', {}).get('text', 'No text detected'))
                print(f"Confidence: {tesseract_confidence:.2f}")
                
                print(f"\n--- Best Result ({best_engine}) ---")
                print(best_text)
                print(f"Confidence: {confidence:.2f}")
            else:
                # Print result from selected engine
                text = result.get(engine, {}).get('text', 'No text detected')
                confidence = result.get(engine, {}).get('confidence', 0)
                
                print(f"\n=== Results for {os.path.basename(image_path)} ({engine}) ===")
                print(f"Processing time: {processing_time:.2f} seconds")
                print(text)
                print(f"Confidence: {confidence:.2f}")
            
            # Save text to file if requested
            if save_text:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_file = output_path / f"{base_name}_recognized.txt"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    if engine == "both":
                        f.write(f"=== EasyOCR Result ===\n")
                        f.write(result.get('easyocr', {}).get('text', 'No text detected'))
                        f.write(f"\nConfidence: {easyocr_confidence:.2f}")
                        
                        f.write(f"\n\n=== Tesseract Result ===\n")
                        f.write(result.get('tesseract', {}).get('text', 'No text detected'))
                        f.write(f"\nConfidence: {tesseract_confidence:.2f}")
                        
                        f.write(f"\n\n=== Best Result ({best_engine}) ===\n")
                        f.write(best_text)
                        f.write(f"\nConfidence: {confidence:.2f}")
                    else:
                        f.write(text)
                        f.write(f"\nConfidence: {confidence:.2f}")
                
                logger.info(f"Saved recognized text to {output_file}")
            
            # Save JSON output if requested
            if save_json:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                json_file = output_path / f"{base_name}_result.json"
                
                # Create a serializable result dictionary
                json_result = {
                    "file": os.path.basename(image_path),
                    "processing_time": processing_time,
                    "results": {}
                }
                
                # Add engine results
                if engine in ["both", "easyocr"] and "easyocr" in result:
                    json_result["results"]["easyocr"] = {
                        "text": result["easyocr"].get("text", ""),
                        "confidence": result["easyocr"].get("confidence", 0)
                    }
                
                if engine in ["both", "tesseract"] and "tesseract" in result:
                    json_result["results"]["tesseract"] = {
                        "text": result["tesseract"].get("text", ""),
                        "confidence": result["tesseract"].get("confidence", 0)
                    }
                
                # Add best result if both engines were used
                if engine == "both":
                    json_result["best_engine"] = best_engine
                    json_result["best_text"] = best_text
                    json_result["best_confidence"] = confidence
                
                # Write to file
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(json_result, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved JSON results to {json_file}")
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            all_results[image_path] = {"error": str(e)}
    
    return all_results


def main():
    """Main entry point for the command-line application"""
    parser = argparse.ArgumentParser(description="Bangla Handwriting Recognition System")
    
    parser.add_argument(
        "images", 
        nargs="*", 
        help="Paths to images containing Bangla handwriting"
    )
    
    parser.add_argument(
        "--engine",
        default=config.DEFAULT_OCR_ENGINE,
        choices=["tesseract", "easyocr", "both"],
        help="OCR engine to use"
    )
    
    parser.add_argument(
        "--preprocess",
        default=config.DEFAULT_PREPROCESS_METHOD,
        choices=["auto", "adaptive", "otsu", "local", "denoise", "all"],
        help="Preprocessing method to use"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Directory to save output files (defaults to ./static/results)"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization of preprocessing steps"
    )
    
    parser.add_argument(
        "--save-text",
        action="store_true",
        help="Save recognized text to file"
    )
    
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save results as JSON"
    )
    
    parser.add_argument(
        "--api",
        action="store_true",
        help="Start the API server instead of processing images"
    )
    
    parser.add_argument(
        "--host",
        default=config.API_HOST,
        help="Host for the API server (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=config.API_PORT,
        help="Port for the API server (default: 8000)"
    )
    
    args = parser.parse_args()
    
    # Handle API server option
    if args.api:
        import uvicorn
        from app.api import app
        
        logger.info(f"Starting API server at {args.host}:{args.port}...")
        uvicorn.run("app.api:app", host=args.host, port=args.port, reload=True)
    else:
        # Ensure images are provided if not in API mode
        if not args.images:
            parser.print_help()
            print("\nError: At least one image path is required when not in API mode.")
            sys.exit(1)
        
        # Process images
        process_images(
            image_paths=args.images,
            engine=args.engine,
            preprocess=args.preprocess,
            output_dir=args.output_dir,
            visualize=args.visualize,
            save_text=args.save_text,
            save_json=args.save_json
        )

if __name__ == "__main__":
    main()