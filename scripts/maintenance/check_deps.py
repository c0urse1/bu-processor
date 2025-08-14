#!/usr/bin/env python3
"""Check dependencies"""

dependencies = [
    'torch',
    'transformers', 
    'pytest',
    'pydantic',
    'pandas',
    'pinecone',
    'openai',
    'sentence_transformers',
    'pytesseract',
    'fitz'
]

print("Checking dependencies...")
for dep in dependencies:
    try:
        __import__(dep)
        print(f"✅ {dep}")
    except ImportError:
        print(f"❌ {dep} - not installed")
    except Exception as e:
        print(f"⚠️ {dep} - {e}")
