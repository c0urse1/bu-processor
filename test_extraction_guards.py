#!/usr/bin/env python3
"""
Test Guards Against Accidental Re-extraction
=============================================

Verifies that our pipeline guards against accidental re-extraction and
ensures consistent path normalization.
"""

import sys
import logging
from pathlib import Path
from unittest.mock import Mock, patch, call
import tempfile
import os

# Add the project directory to the Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_path_normalization_consistency():
    """Test that path normalization is consistent across different input types."""
    
    with patch('bu_processor.bu_processor.pipeline.enhanced_integrated_pipeline.EnhancedPDFExtractor') as mock_extractor_class:
        with patch('bu_processor.bu_processor.pipeline.enhanced_integrated_pipeline.RealMLClassifier') as mock_classifier_class:
            # Mock the extractor instance
            mock_extractor = Mock()
            mock_extractor.extract_text_from_pdf.return_value = Mock(text="Sample extracted text content")
            mock_extractor_class.return_value = mock_extractor
            
            # Mock the classifier instance
            mock_classifier = Mock()
            mock_classifier_class.return_value = mock_classifier
            
            from bu_processor.bu_processor.pipeline.enhanced_integrated_pipeline import EnhancedIntegratedPipeline
            
            pipeline = EnhancedIntegratedPipeline()
            
            # Test with different path formats
            test_paths = [
                "test.pdf",           # String path
                Path("test.pdf"),     # Path object
                str(Path("test.pdf")) # String from Path
            ]
            
            for test_path in test_paths:
                logger.info(f"Testing path normalization with: {test_path} (type: {type(test_path)})")
                
                # Reset mock calls
                mock_extractor.extract_text_from_pdf.reset_mock()
                
                # Call _extract_text_once with different path types
                try:
                    result_text = pipeline._extract_text_once(test_path)
                    
                    # Verify that extractor was called exactly once
                    mock_extractor.extract_text_from_pdf.assert_called_once()
                    
                    # Get the actual call arguments
                    call_args = mock_extractor.extract_text_from_pdf.call_args
                    actual_path_arg = call_args[0][0]  # First positional argument
                    
                    # Verify path was normalized to string
                    assert isinstance(actual_path_arg, str), f"Path not normalized to string: {type(actual_path_arg)}"
                    assert actual_path_arg == "test.pdf", f"Unexpected normalized path: {actual_path_arg}"
                    
                    logger.info(f"✅ Path normalized correctly: {actual_path_arg}")
                    
                except Exception as e:
                    logger.error(f"❌ Path normalization failed for {test_path}: {e}")
                    raise

def test_validation_guard_without_reextraction():
    """Test that validation uses already extracted text and doesn't call extractor again."""
    
    with patch('bu_processor.bu_processor.pipeline.enhanced_integrated_pipeline.EnhancedPDFExtractor') as mock_extractor_class:
        with patch('bu_processor.bu_processor.pipeline.enhanced_integrated_pipeline.RealMLClassifier') as mock_classifier_class:
            # Mock the extractor instance 
            mock_extractor = Mock()
            mock_extractor_class.return_value = mock_extractor
            
            # Mock the classifier instance
            mock_classifier = Mock()
            mock_classifier_class.return_value = mock_classifier
            
            from bu_processor.bu_processor.pipeline.enhanced_integrated_pipeline import EnhancedIntegratedPipeline, PipelineResult
            
            pipeline = EnhancedIntegratedPipeline()
            result = PipelineResult(input_file="test.pdf", processing_time=0.0, strategy_used="fast")
            
            # Test validation with good text
            good_text = "This is valid extracted text content that should pass validation."
            validation_result = pipeline._validate_pdf_without_reextract(good_text, result)
            
            assert validation_result == True, "Validation should pass for good text"
            assert len(result.errors) == 0, "No errors should be added for valid text"
            
            # Verify extractor was NOT called during validation
            mock_extractor.extract_text_from_pdf.assert_not_called()
            
            logger.info("✅ Validation guard works - no re-extraction during validation")
            
            # Test validation with bad text
            result.errors.clear()  # Reset errors
            bad_text = ""  # Empty text should fail validation
            validation_result = pipeline._validate_pdf_without_reextract(bad_text, result)
            
            assert validation_result == False, "Validation should fail for empty text"
            assert len(result.errors) > 0, "Errors should be added for invalid text"
            
            # Verify extractor was still NOT called
            mock_extractor.extract_text_from_pdf.assert_not_called()
            
            logger.info("✅ Validation correctly rejects bad text without re-extraction")

def test_chunking_uses_extracted_text():
    """Test that chunking uses already extracted text and doesn't re-extract."""
    
    with patch('bu_processor.bu_processor.pipeline.enhanced_integrated_pipeline.EnhancedPDFExtractor') as mock_extractor_class:
        with patch('bu_processor.bu_processor.pipeline.enhanced_integrated_pipeline.RealMLClassifier') as mock_classifier_class:
            # Mock the extractor instance
            mock_extractor = Mock()
            mock_extractor.chunk_text = Mock(return_value=["chunk1", "chunk2", "chunk3"])
            mock_extractor_class.return_value = mock_extractor
            
            # Mock the classifier instance
            mock_classifier = Mock()
            mock_classifier_class.return_value = mock_classifier
            
            from bu_processor.bu_processor.pipeline.enhanced_integrated_pipeline import (
                EnhancedIntegratedPipeline, PipelineResult, PipelineConfig, ExtractedContent
            )
            from bu_processor.bu_processor.pipeline.pdf_extractor import ChunkingStrategy
            
            pipeline = EnhancedIntegratedPipeline()
            
            # Create a result with already extracted content
            extracted_text = "This is already extracted text content that should be chunked."
            extracted_content = ExtractedContent(
                text=extracted_text,
                file_path="test.pdf",
                page_count=1,
                extraction_method="enhanced",
                chunking_enabled=False,
                chunks=[],
                chunking_method=None,
                metadata={}
            )
            
            result = PipelineResult(
                input_file="test.pdf", 
                processing_time=0.0, 
                strategy_used="fast"
            )
            result.extracted_content = extracted_content
            result.raw_text = extracted_text  # Store raw text for reuse
            
            config = PipelineConfig(
                chunking_strategy=ChunkingStrategy.SIMPLE,
                max_chunk_size=1000,
                overlap_size=100
            )
            
            # Call chunking
            result = pipeline._chunk(config, result)
            
            # Verify chunking was successful
            assert result.chunking_success == True, "Chunking should succeed"
            assert len(result.chunks) > 0, "Chunks should be created"
            
            # Verify that chunk_text was called with the raw text (not re-extracted)
            mock_extractor.chunk_text.assert_called_once()
            call_args = mock_extractor.chunk_text.call_args
            actual_text_arg = call_args[0][0]  # First positional argument
            assert actual_text_arg == extracted_text, "chunk_text should use the already extracted text"
            
            # Verify that extract_text_from_pdf was NOT called during chunking
            mock_extractor.extract_text_from_pdf.assert_not_called()
            
            logger.info("✅ Chunking correctly uses already extracted text without re-extraction")

def test_single_extraction_pattern():
    """Test that the entire pipeline calls extractor only once."""
    
    # Create a temporary PDF file for testing
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        temp_file.write(b'%PDF-1.4\n%dummy pdf content\n%%EOF\n')
        temp_pdf_path = temp_file.name
    
    try:
        with patch('bu_processor.bu_processor.pipeline.enhanced_integrated_pipeline.EnhancedPDFExtractor') as mock_extractor_class:
            with patch('bu_processor.bu_processor.pipeline.enhanced_integrated_pipeline.RealMLClassifier') as mock_classifier_class:
                # Mock the extractor instance
                mock_extractor = Mock()
                mock_extractor.extract_text_from_pdf.return_value = Mock(
                    text="Sample extracted text for full pipeline test"
                )
                mock_extractor.chunk_text = Mock(return_value=["chunk1", "chunk2"])
                mock_extractor_class.return_value = mock_extractor
                
                # Mock the classifier instance
                mock_classifier = Mock()
                mock_classifier.classify_text = Mock(return_value={"category": "test", "confidence": 0.95})
                mock_classifier_class.return_value = mock_classifier
                
                from bu_processor.bu_processor.pipeline.enhanced_integrated_pipeline import EnhancedIntegratedPipeline
                
                pipeline = EnhancedIntegratedPipeline()
                
                # Process the document through the full pipeline
                result = pipeline.process_document(temp_pdf_path, strategy="fast")
                
                # Verify that extraction was successful
                assert result.extraction_success == True, "Extraction should succeed"
                
                # Verify that extractor was called exactly ONCE during the entire pipeline
                mock_extractor.extract_text_from_pdf.assert_called_once()
                
                # Get the call arguments to verify path normalization
                call_args = mock_extractor.extract_text_from_pdf.call_args
                actual_path_arg = call_args[0][0]
                
                # Path should be normalized to string
                assert isinstance(actual_path_arg, str), "Path should be normalized to string"
                
                logger.info("✅ Full pipeline calls extractor exactly once with normalized path")
                
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)

def main():
    """Run all guard tests."""
    
    logger.info("🧪 Testing Guards Against Accidental Re-extraction")
    logger.info("=" * 55)
    
    try:
        test_path_normalization_consistency()
        logger.info("✅ Path normalization test passed")
        
        test_validation_guard_without_reextraction()
        logger.info("✅ Validation guard test passed")
        
        test_chunking_uses_extracted_text()
        logger.info("✅ Chunking reuse test passed")
        
        test_single_extraction_pattern()
        logger.info("✅ Single extraction pattern test passed")
        
        logger.info("=" * 55)
        logger.info("🎉 All guard tests passed successfully!")
        logger.info("   - Path normalization is consistent")
        logger.info("   - Validation doesn't re-extract")
        logger.info("   - Chunking reuses extracted text")
        logger.info("   - Full pipeline extracts only once")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()
