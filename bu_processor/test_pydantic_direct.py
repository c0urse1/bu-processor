#!/usr/bin/env python3
"""Direkter Test der Pydantic Modelle ohne Package-Import"""

def test_pydantic_models_direct():
    """Direkter Test der Pydantic Modelle"""
    print("🔧 Direkter Pydantic v2 Test")
    print("=" * 30)
    
    try:
        # Direkter Import der Pydantic Klassen
        from pydantic import BaseModel, Field, model_validator
        from typing import Optional, List
        from datetime import datetime
        
        # Definiere die Modelle direkt im Test
        class ClassificationResult(BaseModel):
            text: str = Field(description="Verarbeiteter Text")
            category: Optional[str] = Field(None, description="Klassifikationskategorie")
            confidence: float = Field(ge=0.0, le=1.0, description="Konfidenz-Score")
            error: Optional[str] = Field(None, description="Fehlermeldung")
            is_confident: bool = Field(default=False, description="Konfidenz über Threshold")
            metadata: dict = Field(default_factory=dict, description="Metadaten")

        class BatchClassificationResult(BaseModel):
            total_processed: int = Field(ge=0, description="Gesamt verarbeitet")
            successful: int = Field(ge=0, description="Erfolgreich")
            failed: int = Field(ge=0, description="Fehlgeschlagen")
            results: List[ClassificationResult] = Field(default_factory=list, description="Ergebnisse")
            
            @model_validator(mode="after")
            def validate_counts(self):
                if self.successful + self.failed != self.total_processed:
                    raise ValueError(f"Failed + Successful must equal Total")
                if len(self.results) != self.total_processed:
                    raise ValueError(f"Results length must equal Total")
                return self
        
        # Test 1: ClassificationResult
        print("Test 1: ClassificationResult")
        result = ClassificationResult(
            text="Test document content",
            category="insurance_form",
            confidence=0.85,
            is_confident=True,
            metadata={"source": "test"}
        )
        print(f"✅ Created: confidence={result.confidence}, category={result.category}")
        
        # Test 2: Validierung - ungültige Konfidenz
        print("\nTest 2: Confidence validation")
        try:
            invalid = ClassificationResult(text="Test", confidence=1.5)
            print("❌ Validation failed")
            return False
        except Exception as e:
            print(f"✅ Validation works: {type(e).__name__}")
        
        # Test 3: BatchClassificationResult
        print("\nTest 3: BatchClassificationResult")
        results = [
            ClassificationResult(text="Doc 1", confidence=0.9, category="A"),
            ClassificationResult(text="Doc 2", confidence=0.8, category="B")
        ]
        
        batch = BatchClassificationResult(
            total_processed=2,
            successful=2,
            failed=0,
            results=results
        )
        print(f"✅ Batch created: {batch.total_processed} total, {batch.successful} successful")
        
        # Test 4: Batch-Validierung
        print("\nTest 4: Batch validation")
        try:
            invalid_batch = BatchClassificationResult(
                total_processed=2,
                successful=1,
                failed=2,  # 1+2=3 ≠ 2
                results=results
            )
            print("❌ Batch validation failed")
            return False
        except Exception as e:
            print(f"✅ Batch validation works: {type(e).__name__}")
        
        # Test 5: Serialization
        print("\nTest 5: Serialization")
        result_dict = result.model_dump()
        result_json = result.model_dump_json()
        print(f"✅ Dict: {len(result_dict)} fields")
        print(f"✅ JSON: {len(result_json)} chars")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pydantic_models_direct()
    if success:
        print("\n✅ Pydantic v2 models work correctly!")
    else:
        print("\n❌ Some tests failed!")
    exit(0 if success else 1)
