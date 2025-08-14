#!/usr/bin/env python3
"""
B) Batch-Counts konsistent - Pytest Version
==========================================
"""

import pytest


class TestBatchCountsConsistent:
    """Test dass BatchClassificationResult Zählungen immer konsistent sind"""
    
    def test_batch_counts_consistent(self):
        """Original Test aus den Requirements"""
        from bu_processor.pipeline.classifier import BatchClassificationResult, ClassificationResult
        
        results = [
            ClassificationResult(
                text="a", 
                category="x", 
                confidence=0.9,
                error=None,
                is_confident=True,
                metadata={}
            ),
            ClassificationResult(
                text="b", 
                category=None, 
                confidence=0.0, 
                error="boom",
                is_confident=False,
                metadata={}
            ),
        ]
        
        model = BatchClassificationResult(
            total_processed=2,
            successful=1,
            failed=1,
            results=results
        )
        
        assert model.total_processed == len(results)
        assert model.successful + model.failed == model.total_processed
    
    def test_batch_counts_with_sanity_guards(self):
        """Test Batch-Counts mit Sanity-Guards Integration"""
        from bu_processor.pipeline.classifier import RealMLClassifier, ClassificationResult
        
        clf = RealMLClassifier.__new__(RealMLClassifier)
        clf.confidence_threshold = 0.7
        
        def mock_classify_text(text):
            if "error" in text:
                raise ValueError("Test error")
            return ClassificationResult(
                text=text,
                category="test",
                confidence=0.8,
                error=None,
                is_confident=True,
                metadata={}
            )
        
        clf.classify_text = mock_classify_text
        
        texts = ["text1", "error_text", "text3", "error_again", "text5"]
        batch_result = clf.classify_batch(texts)
        
        # KONSISTENZ durch Sanity-Guards garantiert
        assert batch_result.total_processed == len(texts)
        assert batch_result.successful + batch_result.failed == batch_result.total_processed
        assert len(batch_result.results) == len(texts)
        
        # Verifikation der Zählungen
        actual_successful = sum(1 for r in batch_result.results if r.error is None)
        actual_failed = sum(1 for r in batch_result.results if r.error is not None)
        
        assert batch_result.successful == actual_successful
        assert batch_result.failed == actual_failed
    
    def test_batch_counts_edge_cases(self):
        """Test Edge-Cases für Batch-Counts"""
        from bu_processor.pipeline.classifier import BatchClassificationResult, ClassificationResult
        
        # Test: Leere Results
        empty_batch = BatchClassificationResult(
            total_processed=0,
            successful=0,
            failed=0,
            results=[]
        )
        
        assert empty_batch.total_processed == len(empty_batch.results)
        assert empty_batch.successful + empty_batch.failed == empty_batch.total_processed
        
        # Test: Alle erfolgreich
        all_success_results = [
            ClassificationResult(
                text=f"success_{i}",
                category="class_a",
                confidence=0.9,
                error=None,
                is_confident=True,
                metadata={}
            ) for i in range(3)
        ]
        
        all_success_batch = BatchClassificationResult(
            total_processed=3,
            successful=3,
            failed=0,
            results=all_success_results
        )
        
        assert all_success_batch.total_processed == len(all_success_results)
        assert all_success_batch.successful + all_success_batch.failed == 3
        assert all_success_batch.failed == 0
    
    def test_batch_counts_with_fixtures(self, test_confidence_threshold):
        """Test Batch-Counts mit Test-Fixtures"""
        from bu_processor.pipeline.classifier import RealMLClassifier, ClassificationResult
        
        clf = RealMLClassifier.__new__(RealMLClassifier)
        clf.confidence_threshold = 0.7  # From fixture
        
        # Deterministic mock
        def predictable_classify_text(text):
            if len(text) % 2 == 0:  # Even length = error
                raise RuntimeError(f"Even length error: {text}")
            
            return ClassificationResult(
                text=text,
                category="odd_length",
                confidence=0.8,
                error=None,
                is_confident=True,
                metadata={"length": len(text)}
            )
        
        clf.classify_text = predictable_classify_text
        
        # Mix von even/odd length texts
        texts = ["a", "bb", "ccc", "dddd", "eeeee"]  # 3 odd, 2 even
        batch_result = clf.classify_batch(texts)
        
        # Konsistenz-Validierung
        assert batch_result.total_processed == len(texts)
        assert batch_result.successful + batch_result.failed == len(texts)
        assert len(batch_result.results) == len(texts)
        
        # Erwartete Verteilung: 3 successful (odd), 2 failed (even)
        assert batch_result.successful == 3
        assert batch_result.failed == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
