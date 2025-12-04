"""
Quick setup verification script.
Tests M4 GPU (MPS) and Gemini API.
"""
import torch
import google.generativeai as genai
from dotenv import load_dotenv
import os

print("=" * 60)
print("GenAI Feedback Summarizer - Setup Verification")
print("=" * 60)

# Test 1: Check PyTorch MPS (M4 GPU) Support
print("\n1. Testing M4 GPU Support...")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"   ✅ M4 GPU (MPS) is available!")
    print(f"   Device: {device}")
    
    # Quick tensor test
    x = torch.randn(1000, 1000, device=device)
    y = torch.matmul(x, x)
    print(f"   ✅ GPU tensor operations working")
else:
    print(f"   ❌ MPS not available (will use CPU)")
    device = torch.device("cpu")

# Test 2: Check Gemini API
print("\n2. Testing Gemini API...")
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    print(f"   ✅ Gemini API key found: {api_key[:20]}...")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')  # Using stable v1 model
        
        # Test API call
        response = model.generate_content("Say 'Hello! Gemini API is working!' in one sentence.")
        print(f"   ✅ Gemini API Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Gemini API error: {e}")
else:
    print(f"   ❌ Gemini API key not found in .env")

# Test 3: Check installed packages
print("\n3. Checking Key Dependencies...")
try:
    import transformers
    print(f"   ✅ transformers: {transformers.__version__}")
except:
    print(f"   ❌ transformers not installed")

try:
    import datasets
    print(f"   ✅ datasets: OK")
except:
    print(f"   ❌ datasets not installed")

try:
    import evaluate
    print(f"   ✅ evaluate: OK")
except:
    print(f"   ❌ evaluate not installed")

print("\n" + "=" * 60)
print("Setup Verification Complete!")
print("=" * 60)

# Summary
print("\n🎯 Next Steps:")
if torch.backends.mps.is_available() and api_key:
    print("   ✅ All systems ready!")
    print("   ✅ You can start Sprint 1 now")
    print("   🚀 Run: python src/train_summarizer.py")
else:
    if not torch.backends.mps.is_available():
        print("   ⚠️  MPS not available - training will be slower on CPU")
    if not api_key:
        print("   ⚠️  Set up .env file with GEMINI_API_KEY")
