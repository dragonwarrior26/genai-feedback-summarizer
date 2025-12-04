# GenAI Feedback Summarizer

## 📌 Project Overview
Briefly describe what this project does.
- **Goal**: Summarize feedback and classify sentiment/topics.
- **Models**: BART/T5 for summarization, RoBERTa/DeBERTa for classification.
- **Input**: Customer feedback text.
- **Output**: Concise summary + Category/Sentiment labels.

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/genai-feedback-summarizer.git
cd genai-feedback-summarizer
```

### 2. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download Models
Please refer to `download_models.txt` for links to the pre-trained models.
Place them in the `models/` directory.

## 🚀 Usage

### 1. Data Preprocessing
Explain how to prepare the data.
```bash
python scripts/preprocess_data.py --input data_raw/feedback.csv --output data_processed/
```

### 2. Running Inference (Summarization & Classification)
```bash
python src/inference.py --text "The product is great but delivery was slow."
```

### 3. Batch Processing
```bash
python src/batch_inference.py --input data_processed/test_set.csv --output outputs/results.csv
```

## 📊 Evaluation
Describe how to reproduce the evaluation metrics.
```bash
python src/evaluate.py --ground_truth data_processed/test.csv --predictions outputs/results.csv
```
- **Summarization**: ROUGE-1, ROUGE-2, ROUGE-L
- **Classification**: Accuracy, F1-Score

## 📂 Project Structure
```
├── data_raw/          # Original dataset
├── data_processed/    # Cleaned data
├── docs/              # Documentation & Reports
├── models/            # Downloaded model weights (not in git)
├── notebooks/         # Jupyter notebooks for experiments
├── outputs/           # Generated summaries & logs
├── scripts/           # Utility scripts (preprocessing, training)
├── src/               # Source code (model definitions, inference)
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## 📜 License
[MIT / Apache 2.0]
