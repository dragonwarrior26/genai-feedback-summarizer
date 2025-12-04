# GenAI Feedback Summarizer API

REST API for programmatic access to feedback summarization and classification.

## Quick Start

```bash
# Start the API server
uvicorn api.main:app --reload --port 8000

# Or run directly
python api/main.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### 1. `/summarize` (POST)
Generate a summary for feedback text.

**Request:**
```json
{
  "text": "The product quality is amazing but delivery was slow.",
  "use_hybrid": false
}
```

**Response:**
```json
{
  "summary": "Product quality is great but delivery was slow.",
  "insights": null,
  "method": "bart_only"
}
```

### 2. `/classify` (POST)
Classify the emotion in feedback.

**Request:**
```json
{
  "text": "I love this product! It exceeded my expectations."
}
```

**Response:**
```json
{
  "label": "Admiration",
  "label_id": 0,
  "confidence": 0.95
}
```

### 3. `/analyze` (POST)
Complete analysis (summarization + classification).

**Request:**
```json
{
  "text": "Great product but slow delivery.",
  "use_hybrid": true
}
```

**Response:**
```json
{
  "summary": "Product quality received positive feedback...",
  "insights": "1. Leverage product quality...",
  "classification": {
    "label": "Admiration",
    "label_id": 0,
    "confidence": 0.92
  },
  "method": "hybrid"
}
```

### 4. `/batch` (POST)
Batch analysis of multiple texts.

**Request:**
```json
{
  "texts": [
    "Great product!",
    "Terrible service."
  ],
  "use_hybrid": false
}
```

**Response:**
```json
{
  "results": [...],
  "count": 2
}
```

## Example Usage

### Python
```python
import requests

url = "http://localhost:8000/analyze"
data = {
    "text": "The product is amazing!",
    "use_hybrid": True
}

response = requests.post(url, json=data)
print(response.json())
```

### cURL
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Great product!", "use_hybrid": true}'
```

### JavaScript
```javascript
fetch('http://localhost:8000/analyze', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    text: 'Amazing quality!',
    use_hybrid: true
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

## Features

- ✅ RESTful API with FastAPI
- ✅ Auto-generated Swagger documentation
- ✅ Hybrid mode (BART + Gemini)
- ✅ Batch processing support
- ✅ CORS enabled
- ✅ Pydantic validation

## Models

- **Summarizer**: BART (fine-tuned on Amazon reviews)
- **Classifier**: RoBERTa (8 emotion classes)
- **LLM**: Gemini 2.5 Flash (optional, for hybrid mode)

## Performance

- **BART-only**: ~500ms per request
- **Hybrid mode**: ~2-3s per request
- **Batch**: Processes sequentially

## Notes

- Gemini API key required for hybrid mode (set in `.env`)
- Models loaded at startup (may take 10-20 seconds)
- Runs on MPS (Mac M4 GPU) if available
