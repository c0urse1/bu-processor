"""
Simple test to verify lazy loading control fixtures work correctly.
"""

import pytest
import os


class TestLazyLoadingFixtures:
    """Simple tests to verify the lazy loading fixtures set environment correctly."""

    def test_non_lazy_models_fixture(self, non_lazy_models):
        """Test that non_lazy_models fixture sets BU_LAZY_MODELS=0."""
        assert os.environ.get("BU_LAZY_MODELS") == "0"

    def test_lazy_models_fixture(self, lazy_models):
        """Test that lazy_models fixture sets BU_LAZY_MODELS=1."""
        assert os.environ.get("BU_LAZY_MODELS") == "1"

    def test_default_environment(self):
        """Test that default environment from _base_env sets BU_LAZY_MODELS=0."""
        # This should show the default behavior without specific fixture
        assert os.environ.get("BU_LAZY_MODELS") == "0"
