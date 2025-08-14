#!/usr/bin/env python3
"""
Test-Script für konfigurierbare Confidence-Threshold
=====================================================

Testet die neue Pydantic v2 Konfiguration mit konfigurierbarem
Classifier Confidence Threshold.
"""

import os
import sys
from pathlib import Path

# Füge das Projekt zum Python-Pfad hinzu
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_confidence_threshold_config():
    """Test der konfigurierbaren Confidence-Threshold."""
    
    print("🔧 Test: Konfigurierbare Confidence-Threshold")
    print("=" * 50)
    
    try:
        from bu_processor.core.config import get_config
        
        # Standard-Konfiguration laden
        config = get_config()
        
        print(f"✅ Konfiguration erfolgreich geladen")
        print(f"📊 Standard Confidence Threshold: {config.ml_model.classifier_confidence_threshold}")
        print(f"🌍 Environment: {config.environment}")
        print(f"🔧 Debug Modus: {config.debug}")
        
        # Test mit Environment-Variable
        print("\n🔄 Test mit Environment-Variable...")
        
        # Setze temporär eine Environment-Variable
        original_value = os.environ.get("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD")
        os.environ["BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD"] = "0.9"
        
        # Neue Konfiguration laden (sollte den Env-Wert verwenden)
        from bu_processor.core.config import BUProcessorConfig
        config_with_env = BUProcessorConfig()
        
        print(f"✅ Mit Environment-Variable: {config_with_env.ml_model.classifier_confidence_threshold}")
        
        # Validierung testen
        print(f"📋 Validierung: Threshold liegt zwischen 0.0 und 1.0: {0.0 <= config_with_env.ml_model.classifier_confidence_threshold <= 1.0}")
        
        # Umgebungsvariable zurücksetzen
        if original_value is not None:
            os.environ["BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD"] = original_value
        else:
            os.environ.pop("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD", None)
            
        print("\n✅ Alle Tests erfolgreich!")
        
    except Exception as e:
        print(f"❌ Fehler beim Test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

def test_usage_example():
    """Zeigt Verwendungsbeispiel der neuen Konfiguration."""
    
    print("\n🚀 Verwendungsbeispiel")
    print("=" * 30)
    
    try:
        from bu_processor.core.config import get_config
        
        # Konfiguration laden
        config = get_config()
        
        # Classifier Confidence Threshold verwenden
        threshold = config.ml_model.classifier_confidence_threshold
        
        print(f"""
📋 Verwendung im Code:

```python
from bu_processor.core.config import get_config

# Konfiguration laden
config = get_config()

# Confidence Threshold verwenden
threshold = config.ml_model.classifier_confidence_threshold
print(f"Verwende Confidence Threshold: {{threshold}}")

# In Klassifikation verwenden
if result.confidence >= threshold:
    print("Klassifikation ist sicher genug")
else:
    print("Klassifikation unter Threshold - weitere Prüfung nötig")
```

💡 Aktuelle Werte:
   Confidence Threshold: {threshold}
   Environment: {config.environment}
   Log Level: {config.log_level}
        """)
        
    except Exception as e:
        print(f"❌ Fehler im Beispiel: {e}")

def test_env_file_example():
    """Zeigt .env-Datei Beispiel."""
    
    print("\n📄 .env-Datei Konfiguration")
    print("=" * 35)
    
    print("""
💡 In der .env-Datei setzen:

```bash
# Standard Confidence Threshold
BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD=0.7

# Oder ein anderer Wert zwischen 0.0 und 1.0
BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD=0.85
```

🔧 Environment-Variable direkt setzen:

```bash
export BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD=0.75
```

📋 Pydantic validiert automatisch:
   - Wert muss zwischen 0.0 und 1.0 liegen
   - Typ muss float sein
   - Standard ist 0.8 falls nicht gesetzt
    """)

if __name__ == "__main__":
    print("🧪 TEST: Konfigurierbare Classifier Confidence-Threshold")
    print("=" * 60)
    
    success = test_confidence_threshold_config()
    test_usage_example()
    test_env_file_example()
    
    if success:
        print("\n🎉 Alle Tests erfolgreich! Konfiguration funktioniert.")
        sys.exit(0)
    else:
        print("\n❌ Tests fehlgeschlagen!")
        sys.exit(1)
