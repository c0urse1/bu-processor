#!/usr/bin/env python3
"""Einfacher Test für Confidence Threshold Konfiguration."""

import os
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def simple_test():
    """Einfacher Test der Konfiguration."""
    try:
        # Test 1: Standard-Wert
        from bu_processor.core.config import MLModelConfig
        
        print("🔧 Test 1: Standard-Konfiguration")
        ml_config = MLModelConfig()
        print(f"✅ Standard Confidence Threshold: {ml_config.classifier_confidence_threshold}")
        
        # Test 2: Mit Environment Variable
        print("\n🔧 Test 2: Mit Environment Variable")
        os.environ["BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD"] = "0.9"
        
        ml_config_env = MLModelConfig()
        print(f"✅ Mit Env Variable: {ml_config_env.classifier_confidence_threshold}")
        
        # Test 3: Validierung
        print("\n🔧 Test 3: Validierung")
        print(f"✅ Typ ist float: {type(ml_config_env.classifier_confidence_threshold)}")
        print(f"✅ Wert im gültigen Bereich (0-1): {0 <= ml_config_env.classifier_confidence_threshold <= 1}")
        
        # Cleanup
        os.environ.pop("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD", None)
        
        print("\n🎉 Alle Tests erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_test()
    sys.exit(0 if success else 1)
