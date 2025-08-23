#!/usr/bin/env python3
"""
📊 CLASSIFICATION METADATA ANALYZER
===================================

Analyzes classification metadata in SQLite and Pinecone storage to validate
that predicted_label and predicted_confidence are properly stored for filtering.

Usage:
    python scripts/check_classification_metadata.py
    python scripts/check_classification_metadata.py --filter-label BU_Antrag
    python scripts/check_classification_metadata.py --min-confidence 0.8
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
    from bu_processor.storage.sqlite_store import SQLiteStore
    import structlog
    logger = structlog.get_logger("metadata_analyzer")
except ImportError as e:
    print(f"❌ Failed to import BU-Processor modules: {e}")
    sys.exit(1)

def analyze_sqlite_metadata(filter_label: str = None, min_confidence: float = None):
    """Analyze classification metadata in SQLite storage"""
    print("💾 SQLite Classification Metadata Analysis")
    print("=" * 50)
    
    try:
        config = get_config()
        storage = SQLiteStore()
        
        # Get classification statistics
        stats = storage.get_classification_stats()
        
        print(f"📊 Overall Statistics:")
        print(f"   Total documents: {stats['total_documents']}")
        
        if stats['labels']:
            print(f"   Label distribution:")
            for label, count in sorted(stats['labels'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / stats['total_documents']) * 100
                print(f"      {label}: {count} ({percentage:.1f}%)")
        
        if stats['confidence']:
            conf_stats = stats['confidence']
            print(f"   Confidence statistics:")
            print(f"      Mean: {conf_stats['mean']:.2%}")
            print(f"      Range: {conf_stats['min']:.2%} - {conf_stats['max']:.2%}")
            print(f"      Documents with confidence: {conf_stats['count']}")
        
        # Search with filters
        print(f"\n🔍 Filtered Search Results:")
        if filter_label or min_confidence:
            print(f"   Filters:")
            if filter_label:
                print(f"      Label: {filter_label}")
            if min_confidence:
                print(f"      Min confidence: {min_confidence:.1%}")
        else:
            print(f"   No filters applied (showing recent documents)")
        
        documents = storage.search_documents(
            query="",
            limit=10,
            predicted_label=filter_label,
            min_confidence=min_confidence
        )
        
        if documents:
            print(f"   Found {len(documents)} documents:")
            
            for i, doc in enumerate(documents, 1):
                metadata = doc.get('meta', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                source = doc.get('source', 'unknown')
                created_at = doc.get('created_at', 'unknown')
                
                # Core classification metadata
                predicted_label = metadata.get('predicted_label', 'N/A')
                predicted_confidence = metadata.get('predicted_confidence', 0)
                
                print(f"      {i:2d}. {source}")
                print(f"          📅 Created: {created_at}")
                print(f"          🏷️ Label: {predicted_label}")
                print(f"          📊 Confidence: {predicted_confidence:.2%}")
                
                # Check for detailed classification metadata
                classification = metadata.get('classification', {})
                if classification:
                    model_labels = classification.get('model_labels', [])
                    labels_source = classification.get('model_info', {}).get('labels_source', 'unknown')
                    print(f"          📚 Model labels: {len(model_labels)} available")
                    print(f"          🏷️ Labels source: {labels_source}")
                
                # Check for all scores
                all_scores = classification.get('all_scores', {}) or metadata.get('all_scores', {})
                if all_scores:
                    print(f"          📈 Score distribution:")
                    for label, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
                        print(f"             {label}: {score:.2%}")
                
                print()  # Empty line between documents
        else:
            print(f"   No documents found matching the criteria")
        
        # Test filtering functionality
        print(f"\n🧪 Filter Functionality Tests:")
        
        # Test each unique label
        for label in stats['labels'].keys():
            label_docs = storage.search_documents(predicted_label=label, limit=1)
            print(f"   Label '{label}': {len(label_docs)} documents found")
        
        # Test confidence filtering
        for threshold in [0.5, 0.7, 0.9]:
            conf_docs = storage.search_documents(min_confidence=threshold, limit=1)
            print(f"   Confidence ≥ {threshold:.0%}: {len(conf_docs)} documents found")
            
    except Exception as e:
        print(f"❌ SQLite analysis failed: {e}")
        import traceback
        traceback.print_exc()

def analyze_pinecone_metadata():
    """Analyze classification metadata in Pinecone (if available)"""
    print(f"\n🌲 Pinecone Classification Metadata Analysis")
    print("=" * 50)
    
    try:
        config = get_config()
        
        # Check if Pinecone is enabled
        pinecone_enabled = getattr(config, 'pinecone', {}).get('enabled', False)
        if not pinecone_enabled:
            print("   ℹ️ Pinecone not enabled in configuration")
            return
        
        # Import Pinecone manager
        from bu_processor.pipeline.pinecone_integration import PineconeManager
        
        pinecone_manager = PineconeManager(config)
        
        # TODO: Add Pinecone-specific metadata analysis
        # This would require implementing query functionality in PineconeManager
        print("   ℹ️ Pinecone metadata analysis not yet implemented")
        print("   💡 Pinecone stores the same metadata structure as SQLite")
        
    except ImportError:
        print("   ℹ️ Pinecone integration not available")
    except Exception as e:
        print(f"   ❌ Pinecone analysis failed: {e}")

def validate_metadata_structure():
    """Validate that documents have the expected metadata structure"""
    print(f"\n✅ Metadata Structure Validation")
    print("=" * 40)
    
    try:
        config = get_config()
        storage = SQLiteStore()
        
        # Get some recent documents
        recent_docs = storage.search_documents(query="", limit=5)
        
        if not recent_docs:
            print("   ℹ️ No documents found to validate")
            return
        
        print(f"   Validating {len(recent_docs)} recent documents:")
        
        # Required fields for filtering/analysis
        required_fields = [
            'predicted_label',
            'predicted_confidence'
        ]
        
        # Optional but recommended fields
        recommended_fields = [
            'predicted_category',
            'classification'
        ]
        
        all_valid = True
        
        for i, doc in enumerate(recent_docs, 1):
            metadata = doc.get('meta', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            source = doc.get('source', f'document_{i}')
            print(f"      {i}. {source}")
            
            # Check required fields
            missing_required = []
            for field in required_fields:
                if field not in metadata:
                    missing_required.append(field)
            
            if missing_required:
                print(f"         ❌ Missing required: {missing_required}")
                all_valid = False
            else:
                print(f"         ✅ All required fields present")
            
            # Check recommended fields
            missing_recommended = []
            for field in recommended_fields:
                if field not in metadata:
                    missing_recommended.append(field)
            
            if missing_recommended:
                print(f"         ⚠️ Missing recommended: {missing_recommended}")
            
            # Validate data types
            if 'predicted_confidence' in metadata:
                confidence = metadata['predicted_confidence']
                if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                    print(f"         ❌ Invalid confidence value: {confidence}")
                    all_valid = False
        
        if all_valid:
            print(f"\n   ✅ All documents have valid metadata structure")
        else:
            print(f"\n   ❌ Some documents have metadata issues")
            
    except Exception as e:
        print(f"   ❌ Validation failed: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Analyze classification metadata in storage systems"
    )
    parser.add_argument(
        '--filter-label', 
        help='Filter documents by predicted label'
    )
    parser.add_argument(
        '--min-confidence', 
        type=float,
        help='Minimum confidence threshold (0.0-1.0)'
    )
    parser.add_argument(
        '--skip-pinecone', 
        action='store_true',
        help='Skip Pinecone analysis'
    )
    
    args = parser.parse_args()
    
    # Validate confidence range
    if args.min_confidence is not None and not (0.0 <= args.min_confidence <= 1.0):
        print("❌ min-confidence must be between 0.0 and 1.0")
        sys.exit(1)
    
    print("📊 Classification Metadata Analysis")
    print("=" * 60)
    
    # Analyze SQLite metadata
    analyze_sqlite_metadata(args.filter_label, args.min_confidence)
    
    # Analyze Pinecone metadata
    if not args.skip_pinecone:
        analyze_pinecone_metadata()
    
    # Validate metadata structure
    validate_metadata_structure()
    
    print(f"\n🎉 Metadata analysis completed!")
    print(f"💡 Use filters to test search functionality:")
    print(f"   --filter-label BU_Antrag")
    print(f"   --min-confidence 0.8")

if __name__ == "__main__":
    main()
