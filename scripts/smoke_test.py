#!/usr/bin/env python3
"""
scripts/smoke_test.py

Purpose:
- Quick environment check to ensure required libraries import correctly.
- Prints versions of key packages and a tiny sanity check for torch availability.

Run:
    source .venv/bin/activate    # if not already activated
    python scripts/smoke_test.py
"""
import sys
import importlib

# Packages to test
packages = [
    ("python", None),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("sklearn", "sklearn"),
    ("sentence_transformers", "sentence_transformers"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
]

def print_version(name, module):
    try:
        if module is None:
            print(f"Python: {sys.version.splitlines()[0]}")
            return
        m = importlib.import_module(module)
        ver = getattr(m, "__version__", "unknown")
        print(f"{name}: {ver}")
    except Exception as e:
        print(f"{name}: FAILED to import ({e})")

if __name__ == "__main__":
    print("Running environment smoke test\n" + "-"*40)
    for display, mod in packages:
        print_version(display, mod)

    # Small torch cuda check (if torch imports)
    try:
        import torch
        print("\nTorch CUDA availability:")
        print("  cuda_available:", torch.cuda.is_available())
        try:
            print("  cuda_device_count:", torch.cuda.device_count())
        except Exception:
            pass
    except Exception:
        pass

    print("\nSmoke test completed.")
