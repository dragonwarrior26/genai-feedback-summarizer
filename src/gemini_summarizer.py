#!/usr/bin/env python3
"""
gemini_summarizer.py

Hybrid summarization using BART + Gemini API for enhanced insights.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=GEMINI_API_KEY)

# Prompt templates
PROMPTS = {
    "extractive": """You are a feedback summarization expert. Given the following customer feedback, create a concise summary that captures the key points.

Feedback: {text}

Summary:""",
    
    "abstractive": """You are a feedback analyst. Read the following customer feedback and create a clear, concise summary that:
1. Highlights the main sentiment (positive/negative/mixed)
2. Identifies key themes
3. Notes any specific issues or praise

Feedback: {text}

Summary:""",
    
    "insight_focused": """You are a product manager analyzing customer feedback. Read the following feedback and provide:
1. A brief summary (2-3 sentences)
2. Key insights or actionable items
3. Sentiment assessment

Feedback: {text}

Analysis:"""
}

def summarize_with_gemini(text, prompt_type="abstractive", model_name="gemini-2.5-flash"):
    """
    Summarize text using Gemini API.
    
    Args:
        text: Input text to summarize
        prompt_type: One of 'extractive', 'abstractive', 'insight_focused'
        model_name: Gemini model to use
    
    Returns:
        str: Generated summary
    """
    try:
        model = genai.GenerativeModel(model_name)
        prompt = PROMPTS.get(prompt_type, PROMPTS["abstractive"]).format(text=text)
        
        response = model.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        print(f"Gemini API Error: {e}", file=sys.stderr)
        return None

def hybrid_summarize(text, bart_summary=None, model_name="gemini-2.5-flash"):
    """
    Hybrid summarization: Use BART summary as context for Gemini refinement.
    
    Args:
        text: Original feedback text
        bart_summary: Pre-generated BART summary (optional)
        model_name: Gemini model to use
    
    Returns:
        dict: {
            'summary': final summary,
            'insights': extracted insights,
            'method': 'hybrid' or 'gemini_only'
        }
    """
    try:
        model = genai.GenerativeModel(model_name)
        
        if bart_summary:
            # Hybrid mode: Use BART summary as context
            prompt = f"""You are refining a machine-generated summary of customer feedback.

Original Feedback: {text}

Machine Summary: {bart_summary}

Your task:
1. Improve the summary for clarity and conciseness
2. Add any missing key points
3. Extract actionable insights

Refined Summary:"""
        else:
            # Gemini-only mode
            prompt = PROMPTS["insight_focused"].format(text=text)
        
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        return {
            'summary': result_text,
            'insights': result_text,  # For now, same as summary
            'method': 'hybrid' if bart_summary else 'gemini_only'
        }
    
    except Exception as e:
        print(f"Hybrid summarization error: {e}", file=sys.stderr)
        # Fallback to BART summary if available
        if bart_summary:
            return {
                'summary': bart_summary,
                'insights': '',
                'method': 'bart_fallback'
            }
        return None

def test_gemini():
    """Test Gemini API connectivity."""
    test_text = "This product is amazing! The quality exceeded my expectations. However, the delivery was slower than promised."
    
    print("Testing Gemini API...")
    print(f"Input: {test_text}\n")
    
    for prompt_type in PROMPTS.keys():
        print(f"--- {prompt_type.upper()} ---")
        summary = summarize_with_gemini(test_text, prompt_type)
        if summary:
            print(f"Summary: {summary}\n")
        else:
            print("Failed!\n")
    
    print("--- HYBRID MODE ---")
    bart_summary = "Product quality is great but delivery was slow."
    result = hybrid_summarize(test_text, bart_summary)
    if result:
        print(f"Method: {result['method']}")
        print(f"Summary: {result['summary']}\n")

if __name__ == "__main__":
    test_gemini()
