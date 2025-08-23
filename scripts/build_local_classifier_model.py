#!/usr/bin/env python3
"""
🚀 BUILD LOCAL CLASSIFIER MODEL ARTIFACT
========================================

Creates a minimal valid model artifact for bootstrapping the BU-Processor API.
Downloads HuggingFace model and reconfigures it for BU classification labels.

Usage:
    python scripts/build_local_classifier_model.py
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def build_local_model():
    """Build local classifier model artifact"""
    
    try:
        from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
        print("✅ Transformers library loaded successfully")
    except ImportError:
        print("❌ transformers library not available")
        print("   Install with: pip install transformers")
        sys.exit(1)
    
    # Configuration
    SRC = "deepset/gbert-base"  # German BERT base model
    ART = Path("artifacts/model-v1")
    ART.mkdir(parents=True, exist_ok=True)
    
    # BU-specific labels
    labels = [
        "BU_Bedingungswerk",
        "BU_Antrag", 
        "BU_Risikopruefung",
        "BU_Leitfaden",
        "BU_Fallbeispiel",
        "BU_FAQ",
        "BU_Presse",
        "Sonstiges"
    ]
    
    print("🏗️ Building Local Classifier Model")
    print("=" * 40)
    print(f"Source model: {SRC}")
    print(f"Target artifact: {ART.absolute()}")
    print(f"Labels: {len(labels)} BU categories")
    
    # Create label mappings
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}
    
    print(f"\n📋 Label Configuration:")
    for i, label in enumerate(labels):
        print(f"   {i}: {label}")
    
    try:
        # 1. Load and configure model config
        print(f"\n⚙️ Loading model configuration...")
        config = AutoConfig.from_pretrained(
            SRC, 
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id
        )
        print(f"   ✅ Config loaded with {config.num_labels} labels")
        
        # 2. Load tokenizer
        print(f"\n🔤 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(SRC, use_fast=True)
        print(f"   ✅ Tokenizer loaded: {tokenizer.__class__.__name__}")
        
        # 3. Load model with new classification head
        print(f"\n🧠 Loading model (this may take a moment)...")
        model = AutoModelForSequenceClassification.from_pretrained(
            SRC, 
            config=config
        )
        print(f"   ✅ Model loaded with new classification head")
        print(f"   ⚠️ Classification head randomly initialized (not trained)")
        
        # 4. Save tokenizer
        print(f"\n💾 Saving tokenizer...")
        tokenizer.save_pretrained(ART)
        print(f"   ✅ Tokenizer saved to {ART}")
        
        # 5. Save model
        print(f"\n💾 Saving model...")
        model.save_pretrained(ART)
        print(f"   ✅ Model saved to {ART}")
        
        # 6. Save labels.txt
        print(f"\n🏷️ Creating labels.txt...")
        labels_file = ART / "labels.txt"
        labels_file.write_text("\n".join(labels) + "\n", encoding="utf-8")
        print(f"   ✅ Labels saved to {labels_file}")
        
        # 7. Create artifact metadata
        print(f"\n📄 Creating artifact metadata...")
        metadata = {
            "artifact_version": "v1",
            "source_model": SRC,
            "num_labels": len(labels),
            "labels": labels,
            "created_at": "2025-08-23T00:00:00Z",
            "description": "Bootstrap BU classifier model artifact",
            "training_status": "untrained",
            "notes": "Classification head randomly initialized - requires training for production use"
        }
        
        import json
        metadata_file = ART / "artifact_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Metadata saved to {metadata_file}")
        
        # 8. Verify artifact
        print(f"\n🔍 Verifying artifact structure...")
        required_files = [
            "config.json",
            "tokenizer.json", 
            "tokenizer_config.json",
            "pytorch_model.bin",
            "labels.txt",
            "artifact_metadata.json"
        ]
        
        missing_files = []
        for file in required_files:
            if not (ART / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"   ⚠️ Missing files: {missing_files}")
        else:
            print(f"   ✅ All required files present")
        
        # List created files
        print(f"\n📁 Created files:")
        for file in sorted(ART.glob("*")):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   {file.name}: {size_mb:.1f} MB")
        
        print(f"\n🎉 Local classifier model artifact created successfully!")
        print(f"📍 Location: {ART.absolute()}")
        print(f"\n💡 Next steps:")
        print(f"   1. Test with: python scripts/validate_labels_metadata.py")
        print(f"   2. Start API: python bu_processor/start_api.py")
        print(f"   3. Train model for production accuracy")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to build model artifact: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 BU-Processor Model Artifact Builder")
    print("=" * 50)
    
    success = build_local_model()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
