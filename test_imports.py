#!/usr/bin/env python3
"""
Simple test script to verify import fixes
"""

def test_imports():
    try:
        # Test the semantic chunking enhancement import (syntax error fix)
        from bu_processor.pipeline.semantic_chunking_enhancement import SemanticClusteringEnhancer
        print("✅ SemanticClusteringEnhancer import successful")
        
        # Test the classifier import
        from bu_processor.pipeline.classifier import RealMLClassifier
        print("✅ RealMLClassifier import successful")
        
        # Test the enhanced integrated pipeline import (import error fix)
        from bu_processor.pipeline.enhanced_integrated_pipeline import EnhancedIntegratedPipeline
        print("✅ EnhancedIntegratedPipeline import successful")
        
        print("\n🎉 All critical imports are working!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    test_imports()
