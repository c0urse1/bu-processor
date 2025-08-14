# BU-Processor REST API Usage Guide

## 🚀 Quick Start

### 1. Start the API Server

```bash
# Option 1: Using CLI
python -m bu_processor.cli api --host 0.0.0.0 --port 8000

# Option 2: Using startup script
python start_api.py --host 0.0.0.0 --port 8000 --reload

# Option 3: Using uvicorn directly
uvicorn bu_processor.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 📚 API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "development",
  "classifier_loaded": true,
  "features_enabled": {
    "vector_db": false,
    "chatbot": false,
    "cache": true,
    "gpu": true,
    "semantic_clustering": true,
    "semantic_deduplication": true
  },
  "uptime_seconds": 123.45
}
```

### Text Classification

```bash
curl -X POST "http://localhost:8000/classify/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Ich arbeite als Softwareentwickler in einer IT-Firma.",
    "include_confidence": true,
    "include_processing_time": true
  }'
```

**Response:**
```json
{
  "category": 2,
  "category_label": "IT/Software",
  "confidence": 0.95,
  "is_confident": true,
  "processing_time": 0.123,
  "input_type": "text"
}
```

### PDF Classification

```bash
curl -X POST "http://localhost:8000/classify/pdf" \
  -F "file=@document.pdf" \
  -F "chunking_strategy=semantic" \
  -F "max_chunk_size=1000" \
  -F "classify_chunks_individually=false"
```

**Response:**
```json
{
  "category": 1,
  "category_label": "Legal/Insurance",
  "confidence": 0.88,
  "is_confident": true,
  "processing_time": 2.456,
  "input_type": "pdf",
  "file_name": "document.pdf",
  "page_count": 5,
  "text_length": 2340,
  "extraction_method": "pymupdf",
  "chunking_enabled": true
}
```

### Batch Text Classification

```bash
curl -X POST "http://localhost:8000/classify/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Ich bin Arzt im Krankenhaus.",
      "Als Lehrer unterrichte ich Mathematik.",
      "Ich arbeite als Marketing Manager."
    ],
    "batch_id": "batch_001"
  }'
```

**Response:**
```json
{
  "batch_id": "batch_001",
  "total_processed": 3,
  "successful": 3,
  "failed": 0,
  "batch_time": 0.456,
  "results": [
    {
      "category": 3,
      "category_label": "Healthcare",
      "confidence": 0.92,
      "is_confident": true,
      "input_type": "text_batch"
    },
    {
      "category": 4,
      "category_label": "Education",
      "confidence": 0.89,
      "is_confident": true,
      "input_type": "text_batch"
    },
    {
      "category": 5,
      "category_label": "Marketing",
      "confidence": 0.85,
      "is_confident": true,
      "input_type": "text_batch"
    }
  ]
}
```

### Model Information

```bash
curl http://localhost:8000/models/info
```

**Response:**
```json
{
  "model_info": {
    "model_dir": "artifacts/model-v1",
    "model_name": "deepset/gbert-base",
    "device": "cuda:0",
    "labels_available": true,
    "label_count": 10,
    "labels": ["Category_0", "Category_1", "..."],
    "batch_size": 16,
    "max_retries": 2,
    "timeout_seconds": 30.0
  },
  "health": {
    "status": "healthy",
    "model_loaded": true,
    "response_time": 0.045
  },
  "available_labels": ["Category_0", "Category_1", "..."],
  "supported_chunking_strategies": ["none", "simple", "semantic", "hybrid", "balanced"]
}
```

## 🔒 Authentication

For production environments, set an API key:

```bash
export BU_PROCESSOR_API__API_KEY="your-secret-api-key"
```

Then include the API key in requests:

```bash
curl -X POST "http://localhost:8000/classify/text" \
  -H "Authorization: Bearer your-secret-api-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Example text"}'
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
docker build -t bu-processor-api .

# Run container
docker run -p 8000:8000 \
  -e BU_PROCESSOR_ENVIRONMENT=production \
  -e BU_PROCESSOR_API__API_KEY=your-secret-key \
  bu-processor-api
```

### Using Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f bu-processor-api

# Stop services
docker-compose down
```

## 📊 Error Handling

All API errors return a standardized format:

```json
{
  "error": "Text classification failed",
  "error_type": "ClassificationError",
  "detail": "Model not loaded",
  "request_id": "abc123-def456-ghi789"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid input)
- `401`: Unauthorized (missing API key)
- `403`: Forbidden (invalid API key)
- `413`: Payload Too Large (file too big)
- `422`: Validation Error (invalid parameters)
- `500`: Internal Server Error
- `503`: Service Unavailable (classifier not loaded)

## 🔧 Configuration

Environment variables can be set to configure the API:

```bash
# API Settings
export BU_PROCESSOR_API__HOST=0.0.0.0
export BU_PROCESSOR_API__PORT=8000
export BU_PROCESSOR_API__API_KEY=your-secret-key

# ML Model Settings
export BU_PROCESSOR_ML_MODEL__MODEL_PATH=path/to/model
export BU_PROCESSOR_ML_MODEL__USE_GPU=true
export BU_PROCESSOR_ML_MODEL__CONFIDENCE_THRESHOLD=0.8

# PDF Processing Settings
export BU_PROCESSOR_PDF_PROCESSING__MAX_PDF_SIZE_MB=10
export BU_PROCESSOR_PDF_PROCESSING__ENABLE_CACHE=true
```

## 🚀 Performance Tips

1. **Batch Processing**: Use `/classify/batch` for multiple texts to improve throughput
2. **Chunking Strategy**: Use `simple` for faster processing, `semantic` for better accuracy
3. **GPU Acceleration**: Enable GPU if available for faster inference
4. **Caching**: Enable PDF cache to avoid re-processing identical files
5. **Workers**: Use multiple workers in production: `--workers 4`

## 📝 Example Python Client

```python
import requests

class BUProcessorClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def classify_text(self, text):
        response = requests.post(
            f"{self.base_url}/classify/text",
            json={"text": text},
            headers=self.headers
        )
        return response.json()
    
    def classify_pdf(self, pdf_path):
        with open(pdf_path, "rb") as f:
            files = {"file": f}
            response = requests.post(
                f"{self.base_url}/classify/pdf",
                files=files,
                headers=self.headers
            )
        return response.json()
    
    def health_check(self):
        response = requests.get(f"{self.base_url}/health")
        return response.json()

# Usage
client = BUProcessorClient(api_key="your-api-key")
result = client.classify_text("Beispieltext für Klassifikation")
print(f"Kategorie: {result['category']}, Confidence: {result['confidence']}")
```
