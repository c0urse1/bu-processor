#!/usr/bin/env python3
"""
Test für Pydantic v2 Ergebnis-Modelle
====================================

Testet die neuen robusten ClassificationResult und BatchClassificationResult
Modelle mit Pydantic v2 Validierung.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_classification_result():
    """Test ClassificationResult mit Pydantic v2"""
    print("🔧 Test 1: ClassificationResult")
    
    try:
        from bu_processor.pipeline.classifier import ClassificationResult
        
        # Test 1: Gültige Erstellung
        result = ClassificationResult(
            text="Test text",
            category="insurance_document", 
            confidence=0.85,
            is_confident=True,
            metadata={"source": "test"}
        )
        
        print(f"✅ ClassificationResult erstellt: confidence={result.confidence}")
        print(f"✅ Text: {result.text}")
        print(f"✅ Kategorie: {result.category}")
        print(f"✅ Metadata: {result.metadata}")
        
        # Test 2: Validierung - ungültige Konfidenz
        try:
            invalid_result = ClassificationResult(
                text="Test",
                confidence=1.5  # Ungültig: > 1.0
            )
            print("❌ Validierung fehlgeschlagen - ungültige Konfidenz wurde akzeptiert")
            return False
        except Exception as e:
            print(f"✅ Validierung funktioniert: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fehler in ClassificationResult: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_classification_result():
    """Test BatchClassificationResult mit model_validator"""
    print("\n🔧 Test 2: BatchClassificationResult")
    
    try:
        from bu_processor.pipeline.classifier import ClassificationResult, BatchClassificationResult
        
        # Erstelle Test-Ergebnisse
        results = [
            ClassificationResult(text="Text 1", confidence=0.9, category="A"),
            ClassificationResult(text="Text 2", confidence=0.8, category="B"),
            ClassificationResult(text="Text 3", confidence=0.7, error="Test error")
        ]
        
        # Test 1: Gültige Batch-Ergebnisse
        batch = BatchClassificationResult(
            total_processed=3,
            successful=2,
            failed=1,
            results=results
        )
        
        print(f"✅ BatchResult erstellt: {batch.total_processed} total, {batch.successful} successful")
        print(f"✅ Results length: {len(batch.results)}")
        
        # Test 2: Validierung - inkonsistente Zahlen
        try:
            invalid_batch = BatchClassificationResult(
                total_processed=3,
                successful=2,
                failed=2,  # 2 + 2 = 4 ≠ 3 total
                results=results
            )
            print("❌ Validierung fehlgeschlagen - inkonsistente Zahlen wurden akzeptiert")
            return False
        except Exception as e:
            print(f"✅ Validierung funktioniert: {type(e).__name__}")
        
        # Test 3: Validierung - falsche Results-Länge
        try:
            invalid_batch2 = BatchClassificationResult(
                total_processed=5,  # 5 ≠ 3 results
                successful=3,
                failed=2,
                results=results  # Nur 3 Elemente
            )
            print("❌ Validierung fehlgeschlagen - falsche Results-Länge wurde akzeptiert")
            return False
        except Exception as e:
            print(f"✅ Results-Längen-Validierung funktioniert: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fehler in BatchClassificationResult: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pdf_classification_result():
    """Test PDFClassificationResult Vererbung"""
    print("\n🔧 Test 3: PDFClassificationResult")
    
    try:
        from bu_processor.pipeline.classifier import PDFClassificationResult
        
        pdf_result = PDFClassificationResult(
            text="PDF content",
            category="insurance_form",
            confidence=0.92,
            file_path="/path/to/file.pdf",
            page_count=5,
            extraction_method="pymupdf",
            chunking_enabled=True,
            chunking_method="semantic"
        )
        
        print(f"✅ PDFResult erstellt: {pdf_result.file_path}")
        print(f"✅ Pages: {pdf_result.page_count}")
        print(f"✅ Extraction: {pdf_result.extraction_method}")
        print(f"✅ Chunking: {pdf_result.chunking_enabled}")
        
        # Überprüfe Vererbung
        print(f"✅ Vererbung funktioniert: confidence={pdf_result.confidence}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fehler in PDFClassificationResult: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_serialization():
    """Test Serialisierung der Modelle"""
    print("\n🔧 Test 4: Model Serialization")
    
    try:
        from bu_processor.pipeline.classifier import ClassificationResult, BatchClassificationResult
        
        # Test dict() Methode
        result = ClassificationResult(
            text="Test",
            confidence=0.85,
            category="test_category"
        )
        
        result_dict = result.model_dump()  # Pydantic v2 Syntax
        print(f"✅ model_dump() funktioniert: {len(result_dict)} Felder")
        print(f"✅ Enthält 'text': {'text' in result_dict}")
        print(f"✅ Enthält 'confidence': {'confidence' in result_dict}")
        
        # Test JSON Serialisierung
        result_json = result.model_dump_json()
        print(f"✅ JSON Serialisierung: {len(result_json)} Zeichen")
        
        return True
        
    except Exception as e:
        print(f"❌ Fehler in Serialization: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Führe alle Tests aus"""
    print("🧪 TEST: Pydantic v2 Ergebnis-Modelle")
    print("=" * 50)
    
    tests = [
        test_classification_result,
        test_batch_classification_result,
        test_pdf_classification_result,
        test_model_serialization
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print(f"\n📊 Ergebnis: {passed}/{len(tests)} Tests bestanden")
    
    if passed == len(tests):
        print("🎉 Alle Tests erfolgreich! Pydantic v2 Modelle funktionieren.")
        return True
    else:
        print("❌ Einige Tests fehlgeschlagen!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
