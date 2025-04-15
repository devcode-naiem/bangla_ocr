# Complete README.md for Bangla Handwriting Recognition System

# Bangla Handwriting Recognition System

A comprehensive solution for recognizing Bangla (Bengali) handwritten text from images, combining multiple OCR engines and advanced preprocessing techniques to achieve superior recognition accuracy.

## Features

- **Multiple OCR Engines**: Leverages both Tesseract OCR and EasyOCR for optimal results
- **Advanced Preprocessing**: Intelligent image preprocessing optimized for Bangla handwriting
- **Modular Architecture**: Designed with modularity and extensibility in mind
- **Multiple Interfaces**: 
  - Command-line interface for direct usage
  - REST API for integration with other applications
- **Visualization Tools**: Comprehensive visualization of preprocessing and recognition results
- **Confidence Scoring**: Detailed confidence metrics for recognized text

## System Architecture

The system is organized into a modular structure:

```
bangla_ocr/
│
├── app/                     # Application layer
│   ├── main.py              # CLI entry point
│   ├── api.py               # API implementation
│   ├── config.py            # Configuration
│
├── core/                    # Core functionality
│   ├── recognizer.py        # Main recognition engine
│   ├── preprocessing.py     # Image preprocessing
│   ├── models/              # OCR models
│       ├── base.py          # Abstract model interface
│       ├── tesseract_model.py  # Tesseract implementation
│       ├── easyocr_model.py    # EasyOCR implementation
│
├── utils/                   # Utilities
│   ├── image_utils.py       # Image handling
│   ├── visualization.py     # Visualization tools
│
├── static/                  # Static files
│
├── tests/                   # Unit tests
│
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR with Bengali language support
- [Optional] CUDA-compatible GPU for faster EasyOCR processing

### Installing Tesseract OCR

#### macOS
```bash
brew install tesseract
brew install tesseract-lang  # Installs all language packs including Bengali
```

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-ben  # Bengali language pack
```

#### Windows
1. Download and install Tesseract from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Make sure to select Bengali language during installation
3. Add Tesseract to your PATH environment variable

### Installing the Bangla OCR System

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bangla-ocr.git
cd bangla-ocr
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional):
```bash
# Create a .env file with these variables
TESSERACT_PATH=/path/to/tesseract  # Only needed if not in PATH
TESSDATA_PREFIX=/path/to/tessdata  # Path to Tesseract language data
```

## Usage

### Command Line Interface

#### Recognize Text from an Image

```bash
python -m app.main image.png
```

#### Use Specific OCR Engine

```bash
python -m app.main image.png --engine tesseract
python -m app.main image.png --engine easyocr
python -m app.main image.png --engine both
```

#### Apply Specific Preprocessing

```bash
python -m app.main image.png --preprocess otsu
```

#### Save Results

```bash
python -m app.main image.png --save-text --save-json --output-dir ./results
```

#### Process Multiple Images

```bash
python -m app.main image1.png image2.png image3.png --engine both
```

#### Generate Visualizations

```bash
python -m app.main image.png --visualize
```

### API Server

#### Start the Server

```bash
python -m app.main --api
# or with custom host/port
python -m app.main --api --host 127.0.0.1 --port 8080
```

#### API Endpoints

- `POST /api/recognize`: Recognize text from an uploaded image
  - Form data: `file` (image file)
  - Query parameters:
    - `engine`: OCR engine to use (`tesseract`, `easyocr`, or `both`) 
    - `preprocess`: Preprocessing method (`auto`, `adaptive`, `otsu`, `local`, `denoise`, or `all`)
    - `visualize`: Whether to generate visualization (`true` or `false`)

- `GET /api/health`: Check API health
- `GET /api/engines`: Get information about available OCR engines
- `GET /api/visualization/{viz_id}`: Get visualization by ID

#### Example API Request with cURL

```bash
curl -X POST "http://localhost:8000/api/recognize?engine=both&preprocess=auto" \
  -F "file=@image.png" \
  -H "accept: application/json"
```

#### Example API Request with Python

```python
import requests

url = "http://localhost:8000/api/recognize"
params = {"engine": "both", "preprocess": "auto", "visualize": "true"}

with open("image.png", "rb") as f:
    files = {"file": f}
    response = requests.post(url, params=params, files=files)

result = response.json()
print(result)

# If visualization was requested, get the image
if result["success"] and result["result"]["visualization_id"]:
    viz_id = result["result"]["visualization_id"]
    viz_url = f"http://localhost:8000/api/visualization/{viz_id}"
    viz_response = requests.get(viz_url)
    
    # Save the visualization
    with open("visualization.png", "wb") as f:
        f.write(viz_response.content)
```

## API Response Format

```json
{
  "success": true,
  "result": {
    "text": "রবির গান শুনে মুগ্ধ বনচর, রবি আমাদের বাংলাদেশের বুলবুল। একদা বোনো উৎসব উপলক্ষে রবীন্দ্রনাথ অনেকগুলি গান রচনা করেছিলেন। মহষিদের তখন ছিলেন চুচুড়ায়। কাবর ডাক পড়ল সেখানে।",
    "confidence": 0.89,
    "engine": "easyocr",
    "processing_time": 1.25,
    "visualization_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

## Troubleshooting

### Common Issues

#### Tesseract Not Found

Error: `"Error: Tesseract not found. Please ensure it's installed correctly."`

Solution: Make sure Tesseract is installed and in your PATH, or set the `TESSERACT_PATH` environment variable.

#### Bengali Language Data Not Found

Error: `"Error opening data file for language 'ben'"`

Solutions:
1. Ensure the Bengali language pack is installed
2. Set the `TESSDATA_PREFIX` environment variable to point to your Tesseract data directory
3. Manually download the Bengali traineddata file:
   ```bash
   # Create user tessdata directory
   mkdir -p ~/tessdata
   
   # Download Bengali data
   curl -L https://github.com/tesseract-ocr/tessdata/raw/main/ben.traineddata -o ~/tessdata/ben.traineddata
   
   # Set environment variable
   export TESSDATA_PREFIX=~/tessdata
   ```

#### Poor Recognition Results

If the text recognition quality is poor:

1. Try different preprocessing methods:
   ```bash
   python -m app.main image.png --preprocess otsu
   python -m app.main image.png --preprocess local
   python -m app.main image.png --preprocess denoise
   ```

2. Use both OCR engines:
   ```bash
   python -m app.main image.png --engine both
   ```

3. Improve your input image:
   - Ensure good lighting with no shadows
   - Use dark ink on white paper
   - Write characters with clear spaces between them
   - Keep the page flat when taking the photo

## Using as a Library

The system can also be used as a library in your Python code:

```python
from core.recognizer import BanglaRecognizer

# Initialize the recognizer
recognizer = BanglaRecognizer(
    use_tesseract=True,
    use_easyocr=True
)

# Recognize text from an image
result = recognizer.recognize(
    "path/to/image.png",
    preprocess_method="auto",
    visualize=True
)

# Access the results
if 'easyocr' in result:
    print("EasyOCR result:", result['easyocr']['text'])
    print("Confidence:", result['easyocr']['confidence'])

if 'tesseract' in result:
    print("Tesseract result:", result['tesseract']['text'])
    print("Confidence:", result['tesseract']['confidence'])
```

## Performance Considerations

- EasyOCR generally provides better results for handwritten Bengali text but is slower
- Tesseract is faster but may be less accurate for handwriting
- Using GPU acceleration significantly improves EasyOCR performance
- For batch processing, use the command-line interface with multiple images
- API server performance can be improved by enabling result caching

## Advanced Configuration

### Environment Variables

The system supports the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `TESSERACT_PATH` | Path to Tesseract executable | System default |
| `TESSDATA_PREFIX` | Path to Tesseract language data | System default |
| `API_HOST` | Host address for the API server | 0.0.0.0 |
| `API_PORT` | Port for the API server | 8000 |
| `DEFAULT_OCR_ENGINE` | Default OCR engine to use | easyocr |
| `DEFAULT_PREPROCESS_METHOD` | Default preprocessing method | auto |
| `MAX_IMAGE_SIZE` | Maximum image dimension in pixels | 1920 |
| `EASYOCR_GPU` | Whether to use GPU for EasyOCR | False |
| `ENABLE_CACHE` | Whether to enable result caching | True |
| `CACHE_EXPIRATION` | Cache expiration time in seconds | 3600 |

### Custom OCR Model Training

For achieving even better results with specific types of Bangla handwriting, you can train custom models:

1. **Tesseract Training**:
   - Follow the [Tesseract Training Guide](https://tesseract-ocr.github.io/tessdoc/TrainingTesseract-4.00.html)
   - Use your own handwriting samples for training
   - Place the custom traineddata file in your tessdata directory

2. **EasyOCR Custom Models**:
   - Collect a dataset of your handwriting
   - Use the EasyOCR training pipeline to create a custom model
   - Integrate it with the system by updating the model paths

## Deployment

### Docker Deployment

A Dockerfile is provided for easy deployment:

```dockerfile
FROM python:3.9-slim

# Install Tesseract and Bengali language pack
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ben \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for static files
RUN mkdir -p static/uploads static/results

# Expose the API port
EXPOSE 8000

# Run the API server
CMD ["python", "-m", "app.main", "--api", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run the Docker container:

```bash
docker build -t bangla-ocr .
docker run -p 8000:8000 bangla-ocr
```

### Production Deployment

For production deployment:

1. Use a production ASGI server like Uvicorn with Gunicorn:
   ```bash
   gunicorn app.api:app -k uvicorn.workers.UvicornWorker -w 4 --bind 0.0.0.0:8000
   ```

2. Consider setting up a reverse proxy with Nginx or Apache

3. Implement proper authentication and rate limiting for the API

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Check code style
flake8 .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

 
## Acknowledgments

- Thanks to the Tesseract OCR and EasyOCR teams for their excellent OCR engines
- Special thanks to contributors of Bengali language resources
- Inspired by the need for better Bangla handwriting recognition tools

## Citations

If you use this system in your research, please cite:

```bibtex
@software{bangla_ocr_2023,
  author = {Your Name},
  title = {Bangla Handwriting Recognition System},
  year = {2023},
  url = {https://github.com/yourusername/bangla-ocr}
}
```

## Project Roadmap

Future development plans:

- [ ] Implement custom deep learning model specifically for Bangla handwriting
- [ ] Add support for Bangla digit recognition
- [ ] Develop web interface for easier use
- [ ] Add support for document layout analysis
- [ ] Improve handling of complex conjunct consonants (যুক্তাক্ষর)

## Version History

- 0.1.0: Initial release with Tesseract and EasyOCR support
- 0.2.0: Added API server and improved preprocessing
- 0.3.0: Enhanced visualization and result reporting

## Contact

For questions or support, please open an issue on the GitHub repository or contact the project maintainer at your.email@example.com.

## FAQ

### Q: Which OCR engine should I use?

A: For most Bangla handwriting recognition tasks, EasyOCR typically provides better results. However, for printed text or very clear handwriting, Tesseract may be faster. Using both engines with `--engine both` gives you the best of both worlds.

### Q: How can I improve recognition accuracy?

A: Try these approaches:
1. Use different preprocessing methods (otsu, adaptive, local)
2. Ensure good image quality (proper lighting, contrast, no skew)
3. If possible, write with more spacing between characters
4. For specific handwriting styles, consider training a custom model
```

# Getting Started: First-Time Setup Guide

Here's a comprehensive guide to set up and run the Bangla Handwriting Recognition System for the first time:

## 1. Setting Up the Project Structure

```bash
# Create project directory
mkdir bangla-ocr
cd bangla-ocr

# Create main directories
mkdir -p app core core/models utils static static/uploads static/results tests
```

## 2. Installing Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install pytesseract easyocr opencv-python numpy pillow matplotlib
pip install fastapi uvicorn python-multipart pydantic
pip install requests python-dotenv
```

## 3. Installing Tesseract OCR and Bengali Language Support

### On macOS:
```bash
brew install tesseract
brew install tesseract-lang
```

### On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-ben
```

### On Windows:
1. Download the installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer and select Bengali language during installation
3. Add Tesseract to your PATH environment variable

## 4. Verify Tesseract Installation

```bash
# Check if Tesseract is correctly installed
tesseract --version

# Check if Bengali language is available
tesseract --list-langs
```

You should see 'ben' in the list of languages.

## 5. Creating Configuration File

Create a `.env` file in the project root:

```
# Tesseract configuration
TESSERACT_PATH=  # Leave empty if Tesseract is in PATH
TESSDATA_PREFIX=  # Leave empty if using default location

# API configuration
API_HOST=0.0.0.0
API_PORT=8000

# OCR settings
DEFAULT_OCR_ENGINE=both
DEFAULT_PREPROCESS_METHOD=auto
EASYOCR_GPU=False

# Performance settings
ENABLE_CACHE=True
```

## 6. Testing Your First Recognition

1. Place a test image (e.g., `test.png`) with Bangla handwriting in your project folder

2. Run the CLI version:
```bash
python -m app.main test.png --engine both --visualize
```

3. Check the results printed to the console and the visualization saved in the static/results directory

## 7. Starting the API Server

```bash
python -m app.main --api
```

You should see output indicating that the server is running on http://0.0.0.0:8000

## 8. Testing the API

Using curl:
```bash
curl -X POST "http://localhost:8000/api/recognize?engine=both&preprocess=auto" \
  -F "file=@test.png" \
  -H "accept: application/json"
```

Or open your browser and navigate to http://localhost:8000/api/docs to access the interactive API documentation.

## 9. Troubleshooting

If you encounter issues with Bengali language recognition:

```bash
# Create a user tessdata directory
mkdir -p ~/tessdata

# Download Bengali language data file
curl -L https://github.com/tesseract-ocr/tessdata/raw/main/ben.traineddata -o ~/tessdata/ben.traineddata

# Set environment variable
export TESSDATA_PREFIX=~/tessdata
```

Then try running the recognition again:
```bash
python -m app.main test.png
```

## 10. Next Steps

After successfully setting up the system, you can:

1. Experiment with different preprocessing methods
2. Test with various Bangla handwriting samples
3. Compare results between Tesseract and EasyOCR
4. Integrate the API into your applications
5. Customize the system for your specific needs

If you have any questions or need assistance, please refer to the FAQ section in the README or open an issue on the GitHub repository.