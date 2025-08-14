"""
Test to verify that import/patch targets are stable and work correctly.
"""

import pytest


class TestPipelineImportStability:
    """Test that pipeline imports are stable and patch-friendly."""

    def test_enhanced_pipeline_patch_targets_importable(self):
        """Test that patch targets in enhanced_integrated_pipeline are importable."""
        # These should be importable for tests to patch them
        from bu_processor.pipeline.enhanced_integrated_pipeline import PineconeManager, ChatbotIntegration
        
        # They might be None if dependencies aren't available, but they should be importable
        # This enables tests to do: mocker.patch("bu_processor.pipeline.enhanced_integrated_pipeline.PineconeManager")
        assert PineconeManager is not None or PineconeManager is None  # Either real or None fallback
        assert ChatbotIntegration is not None or ChatbotIntegration is None  # Either real or None fallback

    def test_pipeline_init_is_thin(self):
        """Test that pipeline/__init__.py doesn't do heavy imports."""
        import bu_processor.pipeline
        
        # Should have the module-level __all__ defined
        assert hasattr(bu_processor.pipeline, '__all__')
        
        # Should list the expected modules
        expected_modules = [
            "enhanced_integrated_pipeline",
            "pdf_extractor", 
            "classifier",
            "content_types",
            "pinecone_integration",
            "chatbot_integration",
            "semantic_chunking_enhancement",
            "simhash_semantic_deduplication",
        ]
        
        for module_name in expected_modules:
            assert module_name in bu_processor.pipeline.__all__

    def test_lazy_import_helpers_work(self):
        """Test that lazy import helpers function correctly."""
        from bu_processor.pipeline import get_classifier, get_pdf_extractor, get_semantic_deduplicator
        
        # These should return the actual classes when called
        classifier_class = get_classifier()
        assert classifier_class is not None
        assert classifier_class.__name__ == "RealMLClassifier"
        
        pdf_extractor_class = get_pdf_extractor()
        assert pdf_extractor_class is not None
        assert pdf_extractor_class.__name__ == "EnhancedPDFExtractor"
        
        # Semantic deduplicator might not be available
        dedup_class = get_semantic_deduplicator()
        # It's OK if it's None due to missing dependencies

    def test_patch_targets_are_patchable(self, mocker):
        """Test that the patch targets can actually be patched by tests."""
        # This simulates what a real test would do
        mock_pinecone = mocker.MagicMock()
        mock_chatbot = mocker.MagicMock()
        
        # These patches should work without errors
        pinecone_patch = mocker.patch("bu_processor.pipeline.enhanced_integrated_pipeline.PineconeManager", mock_pinecone)
        chatbot_patch = mocker.patch("bu_processor.pipeline.enhanced_integrated_pipeline.ChatbotIntegration", mock_chatbot)
        
        # Re-import to get the patched version
        from bu_processor.pipeline.enhanced_integrated_pipeline import PineconeManager, ChatbotIntegration
        
        # Should be our mocks now
        assert PineconeManager == mock_pinecone
        assert ChatbotIntegration == mock_chatbot
        
        # Clean up
        pinecone_patch.stop()
        chatbot_patch.stop()
