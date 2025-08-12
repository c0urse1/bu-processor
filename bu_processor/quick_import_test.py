#!/usr/bin/env python3
"""
Quick import test for fixed modules
"""
try:
    print("Testing semantic_chunking_enhancement...")
    from bu_processor.pipeline.semantic_chunking_enhancement import SemanticClusteringEnhancer
    print("✅ semantic_chunking_enhancement import OK")
except Exception as e:
    print(f"❌ semantic_chunking_enhancement import failed: {e}")

try:
    print("Testing classifier...")
    from bu_processor.pipeline.classifier import RealMLClassifier
    print("✅ classifier import OK")
except Exception as e:
    print(f"❌ classifier import failed: {e}")

try:
    print("Testing enhanced_integrated_pipeline...")
    from bu_processor.pipeline.enhanced_integrated_pipeline import EnhancedIntegratedPipeline
    print("✅ enhanced_integrated_pipeline import OK")
except Exception as e:
    print(f"❌ enhanced_integrated_pipeline import failed: {e}")

print("Import test complete")
