#!/usr/bin/env python3
"""
📦 MODEL ARTIFACT CREATOR
========================

Creates versioned model artifacts for production deployment.
Downloads HuggingFace models and packages them with labels for offline use.

Usage:
    python scripts/create_model_artifact.py --model deepset/gbert-base --version v1
"""

import os
import sys
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

def create_model_artifact(source_model: str, version: str = "v1", output_dir: str = "artifacts"):
    """
    Create a versioned model artifact from HuggingFace model.
    
    Args:
        source_model: HuggingFace model name (e.g., "deepset/gbert-base")
        version: Version tag (e.g., "v1", "v2")
        output_dir: Output directory for artifacts
    """
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
    except ImportError:
        print("❌ transformers library not available")
        print("   Install with: pip install transformers")
        sys.exit(1)
    
    # Setup paths
    artifacts_dir = Path(output_dir)
    model_dir = artifacts_dir / f"model-{version}"
    
    print(f"📦 Creating Model Artifact")
    print("=" * 40)
    print(f"Source model: {source_model}")
    print(f"Version: {version}")
    print(f"Output directory: {model_dir.absolute()}")
    
    # Create directories
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Download and save model components
    print(f"\n🔄 Downloading model components...")
    
    try:
        # Load model components
        print(f"   📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(source_model, use_fast=True)
        
        print(f"   📥 Loading config...")
        config = AutoConfig.from_pretrained(source_model)
        
        print(f"   📥 Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(source_model)
        
        # Save to local directory
        print(f"\n💾 Saving to artifact directory...")
        
        print(f"   💾 Saving tokenizer...")
        tokenizer.save_pretrained(model_dir)
        
        print(f"   💾 Saving config...")
        config.save_pretrained(model_dir)
        
        print(f"   💾 Saving model...")
        model.save_pretrained(model_dir)
        
        # Create labels.txt file
        labels_file = model_dir / "labels.txt"
        print(f"   📝 Creating labels.txt...")
        
        # Use BU-specific labels
        bu_labels = [
            "BU_Bedingungswerk",
            "BU_Antrag", 
            "BU_Risikopruefung",
            "BU_Leitfaden",
            "BU_Fallbeispiel",
            "BU_FAQ",
            "BU_Presse",
            "Sonstiges"
        ]
        
        with open(labels_file, 'w', encoding='utf-8') as f:
            for label in bu_labels:
                f.write(f"{label}\n")
        
        print(f"   📝 Created labels.txt with {len(bu_labels)} labels")
        
        # Create artifact metadata
        metadata = {
            "version": version,
            "source_model": source_model,
            "created_at": datetime.now().isoformat(),
            "model_type": config.model_type if hasattr(config, 'model_type') else "unknown",
            "num_labels": len(bu_labels),
            "labels": bu_labels,
            "files": os.listdir(model_dir)
        }
        
        metadata_file = model_dir / "artifact_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"   📋 Created artifact metadata")
        
        # Verify artifact
        print(f"\n✅ Verifying artifact...")
        
        # Test loading
        try:
            from bu_processor.pipeline.classifier_loader import load_classifier, get_model_info
            
            model_ref = f"local:{model_dir}"
            info = get_model_info(model_ref)
            
            if info.get("available"):
                print(f"   ✅ Artifact structure valid")
                
                # Try loading
                tok, mdl = load_classifier(model_ref)
                print(f"   ✅ Model loads successfully")
                print(f"   📊 Model type: {info.get('model_type')}")
                print(f"   🏷️  Number of labels: {info.get('num_labels')}")
                
            else:
                print(f"   ❌ Artifact validation failed: {info.get('error')}")
                
        except Exception as e:
            print(f"   ⚠️  Validation failed: {e}")
        
        # Show summary
        print(f"\n🎉 Model Artifact Created Successfully!")
        print(f"📁 Location: {model_dir.absolute()}")
        print(f"📊 Size: {sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")
        print(f"\n🔧 Configuration:")
        print(f"   Set ML_MODEL_REF=local:{model_dir} in your .env file")
        print(f"\n🚀 Blue/Green Deployment:")
        print(f"   Test with: ML_MODEL_REF=local:{model_dir}")
        print(f"   Switch production after validation")
        
        return model_dir
        
    except Exception as e:
        print(f"❌ Failed to create artifact: {e}")
        # Cleanup on failure
        if model_dir.exists():
            shutil.rmtree(model_dir)
        raise

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Create versioned model artifacts")
    parser.add_argument("--model", "-m", default="deepset/gbert-base", 
                       help="HuggingFace model name")
    parser.add_argument("--version", "-v", default="v1",
                       help="Version tag for artifact")
    parser.add_argument("--output", "-o", default="artifacts",
                       help="Output directory for artifacts")
    
    args = parser.parse_args()
    
    try:
        artifact_path = create_model_artifact(
            source_model=args.model,
            version=args.version,
            output_dir=args.output
        )
        print(f"\n✅ Success! Artifact created at: {artifact_path}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
