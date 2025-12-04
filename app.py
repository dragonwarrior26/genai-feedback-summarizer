"""
Streamlit App for GenAI Feedback Summarizer Demo.
"""
import streamlit as st
import torch
import json
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from src.gemini_summarizer import hybrid_summarize

# Page Config
st.set_page_config(
    page_title="GenAI Feedback Summarizer",
    page_icon="🤖",
    layout="wide"
)

# Constants
SUMMARIZER_PATH_V1 = "outputs/summarizer_test"
SUMMARIZER_PATH_V2 = "outputs/summarizer_v2"
CLASSIFIER_PATH_V1 = "outputs/classifier_test_v2"
CLASSIFIER_PATH_V2 = "outputs/classifier_v2"

INSIGHTS_FILE = "outputs/insights/actionable_insights.json"
CLUSTERS_FILE = "outputs/insights/clusters.json"

# Label Mapping for V2 (8 classes)
LABELS_V2 = ["Admiration", "Joy", "Approval", "Anger", "Sadness", "Fear", "Surprise", "Neutral"]

@st.cache_resource
def load_models():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load Summarizer (Prefer V2 if valid)
    sum_path = SUMMARIZER_PATH_V1
    if Path(SUMMARIZER_PATH_V2).exists() and (Path(SUMMARIZER_PATH_V2) / "config.json").exists():
        sum_path = SUMMARIZER_PATH_V2
        
    try:
        sum_tokenizer = AutoTokenizer.from_pretrained(sum_path)
        sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_path).to(device)
        sum_msg = f"Loaded Summarizer from {sum_path}"
    except Exception as e:
        sum_msg = f"Error loading summarizer: {e}"
        sum_model, sum_tokenizer = None, None

    # Load Classifier (Prefer V2 if valid)
    cls_path = CLASSIFIER_PATH_V1
    if Path(CLASSIFIER_PATH_V2).exists() and (Path(CLASSIFIER_PATH_V2) / "config.json").exists():
        cls_path = CLASSIFIER_PATH_V2
        
    is_v2 = (cls_path == CLASSIFIER_PATH_V2)
    try:
        cls_tokenizer = AutoTokenizer.from_pretrained(cls_path)
        cls_model = AutoModelForSequenceClassification.from_pretrained(cls_path).to(device)
        cls_msg = f"Loaded Classifier from {cls_path}"
    except Exception as e:
        cls_msg = f"Error loading classifier: {e}"
        cls_model, cls_tokenizer = None, None

    return (sum_model, sum_tokenizer), (cls_model, cls_tokenizer), device, is_v2, sum_msg, cls_msg

# Load models
(sum_model, sum_tok), (cls_model, cls_tok), device, is_cls_v2, sum_msg, cls_msg = load_models()

# Show status toasts (outside cached function)
if sum_model:
    st.toast(sum_msg)
else:
    st.error(sum_msg)

if cls_model:
    st.toast(cls_msg)
else:
    st.error(cls_msg)

# UI Layout
st.title("🤖 GenAI Feedback Summarizer")
st.markdown("Analyze customer feedback to generate concise summaries and classify sentiment/topics.")

tab1, tab2, tab3 = st.tabs(["📝 Analyze", "📊 Insights", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        text_input = st.text_area("Paste customer feedback here:", height=200, placeholder="e.g., The product quality is amazing, but the delivery took way too long...")
        
        # Model selection
        use_hybrid = st.checkbox("🚀 Use Hybrid Mode (BART + Gemini)", value=False, help="Enhance summaries with Gemini AI for better insights")
        
        if st.button("Analyze Feedback", type="primary"):
            if not text_input:
                st.warning("Please enter some text first.")
            elif sum_model is None or cls_model is None:
                st.error("Models failed to load.")
            else:
                with st.spinner("Analyzing..."):
                    # Summarization
                    inputs = sum_tok(text_input, return_tensors="pt", max_length=1024, truncation=True).to(device)
                    # Use different params based on model version? V2 uses shorter max_length
                    # Ensure decoder_start_token_id is set
                    if sum_model.config.decoder_start_token_id is None:
                        sum_model.config.decoder_start_token_id = sum_tok.bos_token_id

                    summary_ids = sum_model.generate(
                        inputs["input_ids"], 
                        max_length=100, 
                        min_length=10, 
                        length_penalty=2.0, 
                        num_beams=4, 
                        early_stopping=True,
                        decoder_start_token_id=sum_model.config.decoder_start_token_id
                    )
                    bart_summary = sum_tok.decode(summary_ids[0], skip_special_tokens=True)
                    
                    # Hybrid mode: Enhance with Gemini
                    if use_hybrid:
                        try:
                            hybrid_result = hybrid_summarize(text_input, bart_summary)
                            if hybrid_result and hybrid_result['method'] != 'bart_fallback':
                                summary = hybrid_result['summary']
                                insights = hybrid_result.get('insights', '')
                            else:
                                summary = bart_summary
                                insights = ''
                        except Exception as e:
                            st.warning(f"Gemini API unavailable, using BART-only: {e}")
                            summary = bart_summary
                            insights = ''
                    else:
                        summary = bart_summary
                        insights = ''

                    # Classification
                    inputs = cls_tok(text_input, return_tensors="pt", truncation=True, max_length=512).to(device)
                    with torch.no_grad():
                        logits = cls_model(**inputs).logits
                    label_id = logits.argmax().item()
                    
                    # Map Label
                    if is_cls_v2:
                        label_name = LABELS_V2[label_id] if label_id < len(LABELS_V2) else "Unknown"
                    else:
                        label_name = f"Label {label_id}" # V1 didn't have mapping saved

                    # Display Results
                    st.subheader("📄 Summary")
                    if use_hybrid and insights:
                        st.success(summary)
                        with st.expander("💡 AI-Generated Insights", expanded=True):
                            st.markdown(insights)
                    else:
                        st.success(summary)
                    
                    st.subheader("🏷️ Classification")
                    st.info(f"**{label_name}** (ID: {label_id})")

    with col2:
        st.markdown("### Batch Processing")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            st.write("Batch processing not fully connected in UI yet.")

with tab2:
    st.header("🔍 Actionable Insights")
    
    if Path(INSIGHTS_FILE).exists():
        with open(INSIGHTS_FILE, "r") as f:
            insights = json.load(f)
        
        for item in insights:
            with st.expander(f"Cluster {item['cluster_id']}: {item['theme']}", expanded=True):
                st.markdown(f"**Suggestion:** {item['actionable_suggestion']}")
                if item['pain_points']:
                    st.markdown("**Common Pain Points:**")
                    for pp in item['pain_points']:
                        st.markdown(f"- {pp}")
                else:
                    st.markdown("*No specific pain points detected.*")
    else:
        st.warning("No insights generated yet. Run the insight extractor script.")

    st.markdown("---")
    st.subheader("Topic Clusters")
    if Path("outputs/insights/clusters_pca.png").exists():
        st.image("outputs/insights/clusters_pca.png", caption="Feedback Clusters Visualization")
    
    if Path(CLUSTERS_FILE).exists():
        with open(CLUSTERS_FILE, "r") as f:
            clusters = json.load(f)
        st.json(clusters, expanded=False)

with tab3:
    st.header("About")
    st.info("This tool uses BART/T5 for summarization and RoBERTa for classification.")
    st.markdown("""
    **Models:**
    - Summarizer: BART-base (Fine-tuned on Amazon Reviews)
    - Classifier: RoBERTa (Fine-tuned on GoEmotions)
    
    **Features:**
    - Abstractive Summarization
    - Emotion Classification (8 classes)
    - Topic Clustering & Insight Extraction
    """)
    st.write("Created by Aayush Sharma")
