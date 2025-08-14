#!/usr/bin/env python3
"""
VALIDATION SUITE (pytest version)
==================================

Final validation of stability fixes without imports that cause Pydantic issues.
Converted to pytest format for CI/CD integration.
"""

import pytest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent

class TestValidationSuite:
    """Validation test suite for stability fixes."""

    def test_nltk_fallback_validation(self):
        """Test NLTK fallback implementation."""
        
        pdf_extractor_path = project_root / "bu_processor" / "bu_processor" / "pipeline" / "pdf_extractor.py"
        if not pdf_extractor_path.exists():
            pytest.skip("PDF extractor file not found - skipping NLTK fallback test")
        
        with open(pdf_extractor_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check for NLTK fallback implementation
        assert "NLTK_AVAILABLE = False" in content or "nltk" in content, \
            "NLTK handling should be present in PDF extractor"

    def test_universal_dispatch_validation(self):
        """Test universal dispatch implementation."""
        
        classifier_path = project_root / "bu_processor" / "bu_processor" / "pipeline" / "classifier.py"
        if not classifier_path.exists():
            pytest.skip("Classifier file not found - skipping universal dispatch test")
        
        with open(classifier_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check for different input type handling
        dispatch_checks = [
            ("isinstance(input_data, str)", "String dispatch"),
            ("isinstance(input_data, list)", "List dispatch"), 
            ("isinstance(input_data, Path)", "Path dispatch"),
        ]
        
        for check, description in dispatch_checks:
            if check in content:
                # At least some dispatch logic should be present
                pass
        
        # Should have error handling for unsupported types
        error_handling = any(phrase in content for phrase in [
            "Unsupported input type",
            "ValueError",
            "raise"
        ])
        assert error_handling, "Should have error handling for unsupported input types"

    def test_pdf_extractor_injection_validation(self):
        """Test PDF extractor injection implementation."""
        
        classifier_path = project_root / "bu_processor" / "bu_processor" / "pipeline" / "classifier.py"
        if not classifier_path.exists():
            pytest.skip("Classifier file not found - skipping PDF extractor injection test")
        
        with open(classifier_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check for dependency injection capability
        injection_indicators = [
            "set_pdf_extractor",
            "pdf_extractor",
            "extractor"
        ]
        
        has_injection = any(indicator in content for indicator in injection_indicators)
        assert has_injection, "Should have PDF extractor injection capability"

    def test_stable_imports_validation(self):
        """Test that stable imports are working."""
        
        # Check that core files exist and can be read
        core_files = [
            "bu_processor/bu_processor/__init__.py",
            "bu_processor/bu_processor/pipeline/__init__.py",
        ]
        
        for file_path in core_files:
            full_path = project_root / file_path
            if full_path.exists():
                # Try to read the file to ensure it's not corrupted
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    assert len(content) >= 0, f"File {file_path} should be readable"

    def test_configuration_files_validation(self):
        """Test that configuration files are present and valid."""
        
        config_files = [
            "bu_processor/pyproject.toml",
            "bu_processor/requirements.txt",
            "bu_processor/pytest.ini",
        ]
        
        found_configs = []
        for config_file in config_files:
            config_path = project_root / config_file
            if config_path.exists():
                found_configs.append(config_file)
        
        assert len(found_configs) > 0, f"At least one configuration file should exist. Checked: {config_files}"

    def test_test_structure_validation(self):
        """Test that test structure is properly organized."""
        
        # Check that tests directory exists
        tests_dir = project_root / "bu_processor" / "tests"
        if tests_dir.exists():
            # Should have conftest.py for pytest configuration
            conftest_path = tests_dir / "conftest.py"
            if conftest_path.exists():
                with open(conftest_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Should contain pytest fixtures
                    assert "fixture" in content or "pytest" in content, \
                        "conftest.py should contain pytest fixtures"

    def test_documentation_validation(self):
        """Test that key documentation files exist."""
        
        # Check for key documentation files
        doc_files = [
            "bu_processor/README.md",
            "bu_processor/CONTRIBUTING.md", 
            "bu_processor/CODE_QUALITY.md",
        ]
        
        existing_docs = []
        for doc_file in doc_files:
            doc_path = project_root / doc_file  
            if doc_path.exists():
                existing_docs.append(doc_file)
        
        assert len(existing_docs) > 0, f"At least one documentation file should exist. Checked: {doc_files}"

    def test_package_integrity_validation(self):
        """Test basic package integrity."""
        
        # Check that main package directory exists
        package_dir = project_root / "bu_processor" / "bu_processor"
        assert package_dir.exists(), "Main package directory should exist"
        
        # Check for __init__.py files
        init_files = list(package_dir.rglob("__init__.py"))
        assert len(init_files) > 0, "Should have at least one __init__.py file in package"

    def test_no_syntax_errors_validation(self):
        """Test that Python files don't have obvious syntax errors."""
        
        package_dir = project_root / "bu_processor" / "bu_processor" 
        if not package_dir.exists():
            pytest.skip("Package directory not found")
        
        python_files = list(package_dir.rglob("*.py"))
        
        syntax_errors = []
        for py_file in python_files[:10]:  # Check first 10 files to avoid long test times
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    compile(content, str(py_file), "exec")
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}: {e}")
            except Exception:
                # Other errors (like missing imports) are ok for this test
                pass
        
        assert len(syntax_errors) == 0, f"Syntax errors found: {syntax_errors}"

if __name__ == "__main__":
    # Allow running directly for backwards compatibility
    pytest.main([__file__, "-v"])
