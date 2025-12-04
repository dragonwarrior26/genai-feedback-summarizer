"""
FastAPI backend for GenAI Feedback Summarizer.

Provides REST endpoints for summarization and classification.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.gemini_summarizer import hybrid_summarize

app = FastAPI(
    title="GenAI Feedback Summarizer API",
    description="REST API for customer feedback summarization and emotion classification",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
SUMMARIZER_PATH = "outputs/summarizer_test"
CLASSIFIER_PATH = "outputs/classifier_v2"
LABELS = ["Admiration", "Joy", "Approval", "Anger", "Sadness", "Fear", "Surprise", "Neutral"]

# Load models at startup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
sum_tokenizer = None
sum_model = None
cls_tokenizer = None
cls_model = None

@app.on_event("startup")
async def load_models():
    global sum_tokenizer, sum_model, cls_tokenizer, cls_model
    
    print(f"Loading models on {device}...")
    
    # Summarizer
    sum_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_PATH)
    sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_PATH).to(device)
    if sum_model.config.decoder_start_token_id is None:
        sum_model.config.decoder_start_token_id = sum_tokenizer.bos_token_id
    
    # Classifier
    cls_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_PATH)
    cls_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_PATH).to(device)
    
    print("Models loaded successfully!")

# Request/Response models
class FeedbackRequest(BaseModel):
    text: str = Field(..., description="Customer feedback text to analyze")
    use_hybrid: bool = Field(False, description="Use Gemini hybrid mode for enhanced insights")

class SummarizeResponse(BaseModel):
    summary: str
    insights: Optional[str] = None
    method: str

class ClassifyResponse(BaseModel):
    label: str
    label_id: int
    confidence: float

class AnalyzeResponse(BaseModel):
    summary: str
    insights: Optional[str] = None
    classification: ClassifyResponse
    method: str

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., description="List of feedback texts")
    use_hybrid: bool = Field(False, description="Use Gemini hybrid mode")

# Endpoints
@app.get("/")
async def root():
    return {
        "message": "GenAI Feedback Summarizer API",
        "version": "2.0.0",
        "endpoints": {
            "summarize": "/summarize",
            "classify": "/classify",
            "analyze": "/analyze",
            "batch": "/batch"
        }
    }

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: FeedbackRequest):
    """Generate summary for feedback text."""
    try:
        # BART summarization
        inputs = sum_tokenizer(request.text, return_tensors="pt", max_length=1024, truncation=True).to(device)
        summary_ids = sum_model.generate(
            inputs["input_ids"],
            max_length=100,
            min_length=10,
            decoder_start_token_id=sum_model.config.decoder_start_token_id
        )
        bart_summary = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Hybrid mode
        if request.use_hybrid:
            try:
                result = hybrid_summarize(request.text, bart_summary)
                if result and result['method'] != 'bart_fallback':
                    return SummarizeResponse(
                        summary=result['summary'],
                        insights=result.get('insights'),
                        method=result['method']
                    )
            except Exception as e:
                print(f"Gemini error: {e}")
        
        return SummarizeResponse(
            summary=bart_summary,
            insights=None,
            method="bart_only"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: FeedbackRequest):
    """Classify emotion in feedback text."""
    try:
        inputs = cls_tokenizer(request.text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = cls_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            label_id = logits.argmax().item()
            confidence = probs[0][label_id].item()
        
        return ClassifyResponse(
            label=LABELS[label_id],
            label_id=label_id,
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: FeedbackRequest):
    """Complete analysis: summarization + classification."""
    try:
        # Summarize
        sum_response = await summarize(request)
        
        # Classify
        cls_response = await classify(request)
        
        return AnalyzeResponse(
            summary=sum_response.summary,
            insights=sum_response.insights,
            classification=cls_response,
            method=sum_response.method
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
async def batch_analyze(request: BatchRequest):
    """Batch analysis of multiple feedback texts."""
    try:
        results = []
        for text in request.texts:
            req = FeedbackRequest(text=text, use_hybrid=request.use_hybrid)
            result = await analyze(req)
            results.append(result.dict())
        
        return {"results": results, "count": len(results)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
