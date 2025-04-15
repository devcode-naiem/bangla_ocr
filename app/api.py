"""
FastAPI implementation for Bangla Handwriting Recognition API
"""

import os
import uuid
import logging
import time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil

from core.recognizer import BanglaRecognizer
from . import config

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define response models
class OCRResult(BaseModel):
    text: str
    confidence: Optional[float] = None
    engine: str
    processing_time: Optional[float] = None
    visualization_id: Optional[str] = None

class OCRResponse(BaseModel):
    success: bool
    result: Optional[OCRResult] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    versions: Dict[str, str]

class EngineInfo(BaseModel):
    available: bool
    languages: List[str]
    version: Optional[str] = None

class EnginesResponse(BaseModel):
    tesseract: EngineInfo
    easyocr: EngineInfo

# Initialize FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

# Create instance of recognizer to reuse across requests
recognizer = None

# Simple in-memory cache for results
result_cache = {}

def get_cache_key(file_content: bytes, engine: str, preprocess: str) -> str:
    """Generate a cache key based on file content and parameters"""
    import hashlib
    content_hash = hashlib.md5(file_content).hexdigest()
    return f"{content_hash}_{engine}_{preprocess}"

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    global recognizer
    try:
        logger.info("Initializing recognizer...")
        recognizer = BanglaRecognizer(
            use_tesseract=True,
            use_easyocr=True,
            tesseract_path=config.TESSERACT_PATH,
            use_gpu=config.EASYOCR_GPU
        )
        logger.info("Recognizer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize recognizer: {e}")
        # We'll initialize it on demand if startup fails

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    # Clear any temporary files
    try:
        for filename in os.listdir(config.UPLOAD_DIR):
            file_path = os.path.join(config.UPLOAD_DIR, filename)
            if os.path.isfile(file_path) and (time.time() - os.path.getmtime(file_path)) > 3600:
                os.remove(file_path)
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {e}")

def get_recognizer():
    """Get or initialize the recognizer"""
    global recognizer
    if recognizer is None:
        try:
            recognizer = BanglaRecognizer(
                use_tesseract=True,
                use_easyocr=True,
                tesseract_path=config.TESSERACT_PATH,
                use_gpu=config.EASYOCR_GPU
            )
        except Exception as e:
            logger.error(f"Failed to initialize recognizer: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize OCR engine: {e}")
    return recognizer

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint that returns API information"""
    return {
        "name": config.API_TITLE,
        "version": config.API_VERSION,
        "description": config.API_DESCRIPTION,
        "documentation": "/api/docs"
    }

@app.post("/api/recognize", response_model=OCRResponse)
async def recognize_text(
    file: UploadFile = File(...),
    engine: str = Query(config.DEFAULT_OCR_ENGINE, description="OCR engine to use (tesseract, easyocr, or both)"),
    preprocess: str = Query(config.DEFAULT_PREPROCESS_METHOD, description="Preprocessing method to use"),
    visualize: bool = Query(False, description="Whether to generate visualization of preprocessing steps"),
    recognizer_instance: BanglaRecognizer = Depends(get_recognizer)
):
    """
    Recognize Bangla handwriting from an uploaded image
    
    - **file**: Image file containing Bangla handwriting
    - **engine**: OCR engine to use (tesseract, easyocr, or both)
    - **preprocess**: Preprocessing method (auto, adaptive, otsu, local, denoise, all)
    - **visualize**: Whether to generate visualization of preprocessing steps
    """
    # Validate engine parameter
    if engine not in ["tesseract", "easyocr", "both"]:
        return OCRResponse(
            success=False,
            error="Invalid engine parameter. Must be 'tesseract', 'easyocr', or 'both'"
        )
    
    # Validate preprocessing method
    if preprocess not in ["auto", "adaptive", "otsu", "local", "denoise", "all"]:
        return OCRResponse(
            success=False,
            error="Invalid preprocess parameter"
        )
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Check for cached result if caching is enabled
        if config.ENABLE_CACHE:
            cache_key = get_cache_key(file_content, engine, preprocess)
            cached_result = result_cache.get(cache_key)
            
            if cached_result and (time.time() - cached_result.get('timestamp', 0)) < config.CACHE_EXPIRATION:
                logger.info("Using cached result")
                cached_data = cached_result.get('data', {})
                return OCRResponse(
                    success=True,
                    result=OCRResult(**cached_data)
                )
        
        # Create a unique filename
        file_extension = os.path.splitext(file.filename)[1].lower()
        if not file_extension or file_extension not in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            return OCRResponse(
                success=False,
                error="Unsupported file format. Please upload JPG, PNG, BMP, or TIFF images."
            )
            
        temp_file_id = str(uuid.uuid4())
        temp_file = os.path.join(config.UPLOAD_DIR, f"{temp_file_id}{file_extension}")
        
        # Save the uploaded file
        with open(temp_file, "wb") as f:
            f.write(file_content)
        
        # Reset file pointer for future reads
        await file.seek(0)
        
        # Set up OCR parameters
        use_tesseract = engine in ["tesseract", "both"]
        use_easyocr = engine in ["easyocr", "both"]
        
        # Process the image
        logger.info(f"Processing image with engine={engine}, preprocess={preprocess}")
        start_time = time.time()
        
        result = recognizer_instance.recognize(
            temp_file,
            preprocess_method=preprocess,
            use_tesseract=use_tesseract,
            use_easyocr=use_easyocr,
            visualize=visualize
        )
        
        processing_time = time.time() - start_time
        
        # Prepare visualization ID if visualization was generated
        visualization_id = None
        if visualize and result.get('visualization_path'):
            viz_path = result.get('visualization_path')
            if os.path.exists(viz_path):
                viz_id = temp_file_id
                viz_dest = os.path.join(config.RESULT_DIR, f"{viz_id}_viz.png")
                shutil.copy(viz_path, viz_dest)
                visualization_id = viz_id
        
        # Prepare response based on engine selection
        if engine == "both":
            # If using both engines, return the result with highest confidence
            easyocr_confidence = result.get('easyocr', {}).get('confidence', 0)
            tesseract_confidence = result.get('tesseract', {}).get('confidence', 0)
            
            if easyocr_confidence > tesseract_confidence:
                text = result.get('easyocr', {}).get('text', '')
                confidence = easyocr_confidence
                engine_used = "easyocr"
            else:
                text = result.get('tesseract', {}).get('text', '')
                confidence = tesseract_confidence
                engine_used = "tesseract"
        else:
            # Return the result from the selected engine
            text = result.get(engine, {}).get('text', '')
            confidence = result.get(engine, {}).get('confidence', 0)
            engine_used = engine
        
        # Clean up temporary file
        try:
            os.remove(temp_file)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {temp_file}: {e}")
        
        # Prepare result object
        ocr_result = OCRResult(
            text=text,
            confidence=confidence,
            engine=engine_used,
            processing_time=processing_time,
            visualization_id=visualization_id
        )
        
        # Cache the result if caching is enabled
        if config.ENABLE_CACHE:
            result_cache[cache_key] = {
                'timestamp': time.time(),
                'data': ocr_result.dict()
            }
        
        return OCRResponse(
            success=True,
            result=ocr_result
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return OCRResponse(
            success=False,
            error=str(e)
        )

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    engine_versions = {
        "api": config.API_VERSION,
    }
    
    # Add Tesseract version if available
    try:
        import subprocess
        if config.TESSERACT_PATH:
            cmd = [config.TESSERACT_PATH, "--version"]
        else:
            cmd = ["tesseract", "--version"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.strip().split('\n')[0]
            engine_versions["tesseract"] = version_line
    except:
        engine_versions["tesseract"] = "unknown"
    
    # Add EasyOCR version
    try:
        import easyocr
        engine_versions["easyocr"] = easyocr.__version__
    except:
        engine_versions["easyocr"] = "unknown"
    
    return HealthResponse(
        status="ok",
        versions=engine_versions
    )

@app.get("/api/engines", response_model=EnginesResponse)
async def get_available_engines(
    recognizer_instance: BanglaRecognizer = Depends(get_recognizer)
):
    """Get information about available OCR engines"""
    try:
        tesseract_available = recognizer_instance.is_tesseract_available()
        tesseract_languages = recognizer_instance.get_tesseract_languages() if tesseract_available else []
        tesseract_version = "unknown"
        
        # Get Tesseract version
        try:
            import subprocess
            if config.TESSERACT_PATH:
                cmd = [config.TESSERACT_PATH, "--version"]
            else:
                cmd = ["tesseract", "--version"]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                tesseract_version = result.stdout.strip().split('\n')[0]
        except:
            pass
        
        # EasyOCR information
        easyocr_available = recognizer_instance.is_easyocr_available()
        easyocr_version = "unknown"
        
        try:
            import easyocr
            easyocr_version = easyocr.__version__
        except:
            pass
        
        return EnginesResponse(
            tesseract=EngineInfo(
                available=tesseract_available,
                languages=tesseract_languages,
                version=tesseract_version
            ),
            easyocr=EngineInfo(
                available=easyocr_available,
                languages=["bn"],  # EasyOCR supports Bengali by default
                version=easyocr_version
            )
        )
    except Exception as e:
        logger.error(f"Error getting engine information: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/visualization/{viz_id}")
async def get_visualization(viz_id: str):
    """Get visualization by ID"""
    viz_path = os.path.join(config.RESULT_DIR, f"{viz_id}_viz.png")
    if not os.path.exists(viz_path):
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    return FileResponse(viz_path)