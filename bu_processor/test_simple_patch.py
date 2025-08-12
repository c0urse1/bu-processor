"""Simple test to verify patching works."""

def test_patch_pinecone_manager(mocker):
    """Test that PineconeManager can be patched."""
    # Patch in the enhanced_integrated_pipeline where it's actually used
    mock_manager = mocker.patch("bu_processor.pipeline.enhanced_integrated_pipeline.PineconeManager")
    
    # Verify the patch worked
    assert mock_manager is not None
    print("✓ PineconeManager patching works!")

def test_patch_chatbot_integration(mocker):
    """Test that ChatbotIntegration can be patched.""" 
    # Patch in the enhanced_integrated_pipeline where it's aliased/used
    mock_chatbot = mocker.patch("bu_processor.pipeline.enhanced_integrated_pipeline.ChatbotIntegration")
    
    # Verify the patch worked
    assert mock_chatbot is not None
    print("✓ ChatbotIntegration patching works!")

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
