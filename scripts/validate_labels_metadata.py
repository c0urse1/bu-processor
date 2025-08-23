#!/usr/bin/env python3
"""
🏷️ LABEL VALIDATION & METADATA ANALYSIS
=======================================

Validates that labels are loaded correctly from model artifacts and 
analyzes classification metadata in stored documents.

Usage:
    python scripts/validate_labels_metadata.py
    python scripts/validate_labels_metadata.py --check-stored-docs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from bu_processor.core.config import get_config
    from bu_processor.pipeline.classifier import RealMLClassifier
    from bu_processor.pipeline.classifier_loader import load_labels, get_model_info
    from bu_processor.storage.sqlite_store import SqliteStore
    import structlog
    logger = structlog.get_logger("label_validation")
except ImportError as e:
    print(f"❌ Failed to import BU-Processor modules: {e}")
    sys.exit(1)

def validate_artifact_labels():
    """Validate labels in model artifacts"""
    print("🏷️ Label Validation")
    print("=" * 40)
    
    config = get_config()
    model_ref = getattr(config, 'ml_model_ref', None) or getattr(config, 'ML_MODEL_REF', 'local:artifacts/model-v1')
    
    print(f"Model Reference: {model_ref}")
    
    # 1. Test model info
    print(f"\n📊 Model Info:")
    model_info = get_model_info(model_ref)
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # 2. Test label loading
    print(f"\n🏷️ Label Loading:")
    labels = load_labels(model_ref)
    if labels:
        print(f"   ✅ Loaded {len(labels)} labels from artifact")
        print(f"   📋 Labels:")
        for i, label in enumerate(labels, 1):
            print(f"      {i:2d}. {label}")
    else:
        print(f"   ❌ No labels found in artifact")
        return False
    
    # 3. Test classifier integration
    print(f"\n🔄 Classifier Integration:")
    try:
        classifier = RealMLClassifier(config)
        classifier_labels = classifier.get_available_labels()
        
        if classifier_labels:
            print(f"   ✅ Classifier loaded {len(classifier_labels)} labels")
            
            # Compare labels
            if set(labels) == set(classifier_labels):
                print(f"   ✅ Labels match between artifact and classifier")
            else:
                print(f"   ⚠️ Label mismatch detected:")
                print(f"      Artifact: {labels}")
                print(f"      Classifier: {classifier_labels}")
        else:
            print(f"   ❌ Classifier has no labels loaded")
            return False
            
        # Test classification
        test_text = "Dies ist ein Testdokument für die BU-Klassifikation."
        result = classifier.classify_text(test_text)
        
        print(f"\n🧪 Test Classification:")
        print(f"   Text: '{test_text}'")
        print(f"   Predicted: {result['predicted_label']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        
        # Validate predicted label is in artifact labels
        if result['predicted_label'] in labels:
            print(f"   ✅ Predicted label exists in artifact labels")
        else:
            print(f"   ❌ Predicted label NOT in artifact labels!")
            return False
            
    except Exception as e:
        print(f"   ❌ Classifier test failed: {e}")
        return False
    
    return True

def check_stored_document_metadata():
    """Check metadata structure in stored documents"""
    print(f"\n📄 Stored Document Metadata Analysis")
    print("=" * 50)
    
    try:
        config = get_config()
        storage = SqliteStore(config)
        
        # Get recent documents
        recent_docs = storage.search_documents(query="", limit=5)
        
        if not recent_docs:
            print("   ℹ️ No stored documents found")
            return
        
        print(f"   📊 Analyzing {len(recent_docs)} recent documents:")
        
        label_counts = {}
        confidence_scores = []
        metadata_issues = []
        
        for i, doc in enumerate(recent_docs, 1):
            print(f"\n   📄 Document {i}: {doc.get('source', 'unknown')}")
            
            metadata = doc.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            # Check for required classification fields
            required_fields = ['predicted_label', 'predicted_confidence']
            missing_fields = []
            
            for field in required_fields:
                if field not in metadata:
                    missing_fields.append(field)
            
            if missing_fields:
                metadata_issues.append(f"Doc {i}: Missing fields {missing_fields}")
                print(f"      ❌ Missing: {missing_fields}")
            else:
                predicted_label = metadata.get('predicted_label')
                predicted_confidence = metadata.get('predicted_confidence', 0)
                
                print(f"      ✅ Label: {predicted_label}")
                print(f"      ✅ Confidence: {predicted_confidence:.2%}")
                
                # Count labels
                label_counts[predicted_label] = label_counts.get(predicted_label, 0) + 1
                confidence_scores.append(predicted_confidence)
            
            # Check for detailed classification metadata
            classification = metadata.get('classification', {})
            if classification:
                model_labels = classification.get('model_labels', [])
                model_info = classification.get('model_info', {})
                labels_source = model_info.get('labels_source', 'unknown')
                
                print(f"      📊 Model labels count: {len(model_labels)}")
                print(f"      🏷️ Labels source: {labels_source}")
            else:
                print(f"      ⚠️ No detailed classification metadata")
        
        # Summary
        print(f"\n📊 Summary:")
        if label_counts:
            print(f"   Label distribution:")
            for label, count in sorted(label_counts.items()):
                print(f"      {label}: {count}")
        
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            min_confidence = min(confidence_scores)
            max_confidence = max(confidence_scores)
            print(f"   Confidence statistics:")
            print(f"      Average: {avg_confidence:.2%}")
            print(f"      Range: {min_confidence:.2%} - {max_confidence:.2%}")
        
        if metadata_issues:
            print(f"   ⚠️ Metadata issues:")
            for issue in metadata_issues:
                print(f"      {issue}")
        else:
            print(f"   ✅ All documents have required metadata fields")
            
    except Exception as e:
        print(f"   ❌ Failed to analyze stored documents: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Validate labels and metadata in BU-Processor system"
    )
    parser.add_argument(
        '--check-stored-docs', 
        action='store_true',
        help='Also analyze metadata in stored documents'
    )
    
    args = parser.parse_args()
    
    # Validate artifact labels
    success = validate_artifact_labels()
    
    # Check stored documents if requested
    if args.check_stored_docs:
        check_stored_document_metadata()
    
    if success:
        print(f"\n🎉 Label validation completed successfully!")
        if not args.check_stored_docs:
            print(f"💡 Use --check-stored-docs to analyze stored document metadata")
    else:
        print(f"\n❌ Label validation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
