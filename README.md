# 🤖 GenAI Feedback Summarizer

**AI-Powered Customer Feedback Analysis with Hybrid Summarization**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.123-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview

An end-to-end system for analyzing customer feedback using state-of-the-art NLP models. Combines local fine-tuned models (BART, RoBERTa) with cloud LLMs (Gemini) for enhanced summarization and actionable insights.

### Key Features

- 📝 **Hybrid Summarization**: BART + Gemini 2.5 Flash for enhanced summaries
- 🏷️ **Emotion Classification**: 8-class emotion detection (96% improved F1 score)
- 🔍 **Topic Clustering**: Automatic topic discovery with K-Means
- 💡 **Actionable Insights**: AI-generated recommendations from feedback
- 🌐 **REST API**: FastAPI backend with Swagger documentation
- 🎨 **Interactive UI**: Streamlit dashboard with real-time analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- 16GB RAM (for model loading)
- Gemini API key (free tier available)

### Installation

```bash
# Clone repository
git clone https://github.com/dragonwarrior26/genai-feedback-summarizer.git
cd genai-feedback-summarizer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Run Streamlit App

```bash
streamlit run app.py
```

Visit `http://localhost:8501` to access the interactive dashboard.

### Run REST API

```bash
./start_api.sh
# Or manually: .venv/bin/uvicorn api.main:app --reload --port 8000
```

API documentation available at `http://localhost:8000/docs`

## 📊 Models & Performance

### Summarization
- **Model**: BART-base (fine-tuned on Amazon reviews)
- **Enhancement**: Gemini 2.5 Flash (optional hybrid mode)
- **Performance**: BART-only ~500ms, Hybrid ~2-3s

### Classification
- **Model**: RoBERTa-base (8 emotion classes)
- **Accuracy**: 53.5%
- **Macro F1**: 0.51 (96% improvement from v1)
- **Classes**: Admiration, Joy, Approval, Anger, Sadness, Fear, Surprise, Neutral

### Clustering
- **Method**: K-Means (k=5)
- **Embeddings**: all-MiniLM-L6-v2
- **Output**: Topic clusters with keywords and insights

## 🎯 Usage Examples

### Streamlit App

1. **Analyze Tab**: Paste feedback and get instant summary + classification
2. **Insights Tab**: View topic clusters and actionable recommendations
3. **Hybrid Mode**: Toggle for Gemini-enhanced summaries

### REST API

```python
import requests

# Analyze feedback
response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "text": "Great product but delivery was slow!",
        "use_hybrid": True
    }
)

result = response.json()
print(result["summary"])
print(result["classification"]["label"])
```

### Python SDK

```python
from src.inference import summarize_and_classify

text = "Amazing quality, highly recommend!"
summary, label = summarize_and_classify(text)
print(f"Summary: {summary}")
print(f"Emotion: {label}")
```

## 📂 Project Structure

```
genai-feedback-summarizer/
├── api/                    # FastAPI backend
│   ├── main.py            # API endpoints
│   └── README.md          # API documentation
├── src/                    # Core source code
│   ├── gemini_summarizer.py   # Hybrid summarization
│   ├── clustering.py          # Topic clustering
│   ├── insight_extractor.py   # Insight generation
│   ├── train_classifier_v2.py # Classifier training
│   └── inference.py           # Inference utilities
├── scripts/                # Utility scripts
│   ├── eval_summarizer.py     # Model evaluation
│   ├── eval_classifier.py     # Classifier metrics
│   └── pre_demo_check.py      # System verification
├── docs/                   # Documentation & reports
│   ├── class_consolidation_mapping.json
│   ├── rouge_scores.csv
│   └── confusion_matrix.png
├── outputs/                # Model outputs
│   ├── summarizer_test/       # BART model
│   ├── classifier_v2/         # RoBERTa model
│   └── insights/              # Clusters & insights
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Configuration

### Environment Variables

Create `.env` file:

```bash
GEMINI_API_KEY=your_api_key_here
```

### Model Paths

Models are stored in `outputs/`:
- Summarizer: `outputs/summarizer_test/`
- Classifier: `outputs/classifier_v2/`

## 📈 Evaluation

### Run Evaluations

```bash
# Summarizer metrics
python scripts/eval_summarizer.py

# Classifier metrics
python scripts/eval_classifier.py

# System health check
python scripts/pre_demo_check.py
```

### Results

| Metric | Value |
|--------|-------|
| Classifier F1 (v1) | 0.26 |
| Classifier F1 (v2) | **0.51** |
| Improvement | **96%** |
| Clustering Silhouette | 0.31 |
| API Response Time | <3s |

## 🌟 Key Achievements

1. **96% F1 Improvement**: Reduced 28 emotion classes to 8 with class weighting
2. **Cost-Free LLM**: Using Gemini 2.5 Flash (free tier)
3. **Hybrid Architecture**: Combines local + cloud models for best results
4. **Production-Ready API**: FastAPI with auto-generated docs
5. **Actionable Insights**: Automated pain point detection

## 🛠️ Development

### Training Models

```bash
# Train classifier v2
python src/train_classifier_v2.py

# Train summarizer v2 (optional)
python src/train_summarizer_v2.py
```

### Generate Insights

```bash
# Run clustering
python src/clustering.py

# Extract insights
python src/insight_extractor.py
```

## 📚 Documentation

- [API Documentation](api/README.md)
- [Setup Guide](docs/SETUP_GUIDE.md)
- [Project Status](docs/PROJECT_STATUS.md)
- [Implementation Plan](docs/implementation_plan.md)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **Datasets**: Amazon Product Reviews, GoEmotions
- **Models**: Hugging Face Transformers
- **LLM**: Google Gemini API
- **Frameworks**: FastAPI, Streamlit, PyTorch

## 📧 Contact

**Author**: Aayush Sharma  
**GitHub**: [@dragonwarrior26](https://github.com/dragonwarrior26)  
**Project**: [genai-feedback-summarizer](https://github.com/dragonwarrior26/genai-feedback-summarizer)

---

**⭐ Star this repo if you find it useful!**
