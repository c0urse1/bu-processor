#!/usr/bin/env python3
"""
Model Download Script - Production Deployment
============================================

Downloads required models from HuggingFace Hub or cloud storage.
This is the recommended approach for production deployments.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

def download_production_model(
    model_name: str = "deepset/gbert-base", 
    target_dir: str = "artifacts/model-v1",
    force: bool = False
) -> bool:
    """
    Download model for production deployment.
    
    Args:
        model_name: HuggingFace model identifier
        target_dir: Local directory to save model
        force: Overwrite existing model
    
    Returns:
        True if successful, False otherwise
    """
    
    target_path = Path(target_dir)
    
    # Check if model already exists
    if target_path.exists() and not force:
        model_files = list(target_path.glob("*.safetensors"))
        if model_files:
            print(f"✅ Model already exists: {target_path}")
            return True
    
    print(f"📥 Downloading model: {model_name}")
    print(f"🎯 Target directory: {target_path}")
    
    try:
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        
        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Download model components
        print("📦 Downloading model...")
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(target_path)
        
        print("🔤 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(target_path)
        
        print("⚙️ Downloading config...")
        config = AutoConfig.from_pretrained(model_name)
        config.save_pretrained(target_path)
        
        # Create labels file
        labels_path = target_path / "labels.txt"
        if not labels_path.exists():
            print("📝 Creating labels.txt...")
            default_labels = [
                "BU_Bedingungswerk",
                "BU_Antrag", 
                "BU_Risikopruefung",
                "BU_Leitfaden",
                "BU_Fallbeispiel",
                "BU_FAQ",
                "BU_Presse",
                "Sonstiges"
            ]
            
            with open(labels_path, 'w', encoding='utf-8') as f:
                for label in default_labels:
                    f.write(f"{label}\n")
        
        print(f"✅ Model downloaded successfully to {target_path}")
        
        # Show size info
        model_size = sum(f.stat().st_size for f in target_path.rglob('*') if f.is_file())
        print(f"📊 Total model size: {model_size / 1024 / 1024:.1f} MB")
        
        return True
        
    except ImportError:
        print("❌ transformers library not available")
        print("   Install with: pip install transformers")
        return False
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False


def setup_model_for_deployment():
    """Setup model for deployment environment"""
    
    model_ref = os.getenv("ML_MODEL_REF", "local:artifacts/model-v1")
    
    if model_ref.startswith("local:"):
        model_path = model_ref.split("local:", 1)[1]
        
        if not Path(model_path).exists():
            print(f"🚀 Setting up model for first deployment...")
            
            # Try to download default model
            success = download_production_model(
                model_name="deepset/gbert-base",
                target_dir=model_path
            )
            
            if success:
                print("✅ Model setup completed for deployment")
            else:
                print("❌ Model setup failed")
                return False
        else:
            print(f"✅ Model already available: {model_path}")
    
    elif model_ref.startswith("hf:"):
        print(f"✅ Using HuggingFace model: {model_ref}")
    
    else:
        print(f"❌ Unknown model reference: {model_ref}")
        return False
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "setup":
            setup_model_for_deployment()
        elif sys.argv[1] == "download":
            model_name = sys.argv[2] if len(sys.argv) > 2 else "deepset/gbert-base"
            download_production_model(model_name, force=True)
        else:
            print("Usage: python download_model.py [setup|download] [model_name]")
    else:
        setup_model_for_deployment()
