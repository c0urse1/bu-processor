"""Simple test to verify fixtures are working."""

import pytest


def test_sample_texts_fixture(sample_texts):
    """Test that sample_texts fixture works."""
    assert isinstance(sample_texts, list)
    assert len(sample_texts) > 0
    print(f"✓ sample_texts fixture works: {len(sample_texts)} texts")


def test_sample_pdf_path_fixture(sample_pdf_path):
    """Test that sample_pdf_path fixture works."""
    assert sample_pdf_path is not None
    print(f"✓ sample_pdf_path fixture works: {sample_pdf_path}")


def test_classifier_with_mocks_fixture(classifier_with_mocks):
    """Test that classifier_with_mocks fixture works."""
    assert classifier_with_mocks is not None
    # Test that it has the expected methods
    assert hasattr(classifier_with_mocks, 'classify_text')
    assert hasattr(classifier_with_mocks, 'classify_batch')
    print("✓ classifier_with_mocks fixture works")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
