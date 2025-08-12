# Lazy Loading vs from_pretrained Assertions - Solution Guide

## Problem

Tests that expect calls to `AutoTokenizer.from_pretrained()` and `AutoModel.from_pretrained()` may fail when lazy loading is enabled, because these methods are not called immediately during classifier initialization but deferred until the model is actually used.

## Solutions Overview

### 1. Use `disable_lazy_loading` Fixture (Recommended)

The cleanest approach for tests that need immediate `from_pretrained` calls:

```python
def test_classifier_initialization(self, mocker, disable_lazy_loading):
    """Test with lazy loading disabled."""
    mock_tokenizer_patch = mocker.patch(
        "bu_processor.pipeline.classifier.AutoTokenizer.from_pretrained", 
        return_value=mock_tokenizer
    )
    mock_model_patch = mocker.patch(
        "bu_processor.pipeline.classifier.AutoModelForSequenceClassification.from_pretrained", 
        return_value=mock_model
    )
    
    classifier = RealMLClassifier()
    
    # These assertions now work reliably
    mock_tokenizer_patch.assert_called_once()
    mock_model_patch.assert_called_once()
```

### 2. Use `classifier_with_eager_loading` Fixture

Pre-configured fixture with lazy loading disabled:

```python
def test_something(self, classifier_with_eager_loading):
    """Test with pre-configured eager loading classifier."""
    classifier = classifier_with_eager_loading
    
    # Mock references are attached to the classifier
    mock_tokenizer_patch = classifier._test_mock_tokenizer_patch
    mock_model_patch = classifier._test_mock_model_patch
    
    mock_tokenizer_patch.assert_called_once()
    mock_model_patch.assert_called_once()
```

### 3. Manual Loading with Utility Function

For existing tests that use `classifier_with_mocks`:

```python
def test_something(self, mocker, classifier_with_mocks):
    """Test with manual model loading."""
    classifier = classifier_with_mocks
    
    # Set up patches
    mock_tokenizer_patch = mocker.patch("...AutoTokenizer.from_pretrained")
    mock_model_patch = mocker.patch("...AutoModel.from_pretrained")
    
    # Explicitly trigger loading
    from tests.conftest import force_model_loading
    force_model_loading(classifier)
    
    # Or call the method directly
    classifier._load_model_and_tokenizer()
    
    # Now assertions work
    mock_tokenizer_patch.assert_called_once()
    mock_model_patch.assert_called_once()
```

### 4. Direct Monkeypatch in Test

For one-off cases:

```python
def test_something(self, mocker, monkeypatch):
    """Test with direct lazy loading control."""
    # Disable lazy loading for this test
    monkeypatch.setenv("BU_LAZY_MODELS", "0")
    
    # Rest of test...
```

### 5. Factory Function Approach

Programmatic creation:

```python
def test_something(self, mocker):
    """Test with factory function."""
    from tests.conftest import create_eager_classifier_fixture
    
    classifier, mock_tok_patch, mock_model_patch = create_eager_classifier_fixture(mocker)
    
    mock_tok_patch.assert_called_once()
    mock_model_patch.assert_called_once()
```

## Environment Variable Control

The behavior is controlled by the `BU_LAZY_MODELS` environment variable:

- `BU_LAZY_MODELS=1` (default): Lazy loading enabled - models loaded on first use
- `BU_LAZY_MODELS=0`: Lazy loading disabled - models loaded immediately during initialization

## When to Use Each Approach

### Use `disable_lazy_loading` fixture when:
- Testing classifier initialization
- Need to assert on `from_pretrained` calls immediately
- Want clean, declarative test setup

### Use `classifier_with_eager_loading` fixture when:
- Need a complete classifier setup with eager loading
- Want access to mock objects for complex assertions
- Testing multiple aspects of classifier behavior

### Use manual loading when:
- Working with existing tests that use `classifier_with_mocks`
- Need lazy loading for most of the test but eager loading for specific parts
- Debugging loading issues

### Use direct monkeypatch when:
- One-off test that needs special configuration
- Testing environment variable behavior itself
- Quick fixes to existing tests

## Configuration in conftest.py

The solution adds these fixtures to `tests/conftest.py`:

```python
@pytest.fixture
def disable_lazy_loading(monkeypatch):
    """Disable lazy loading for from_pretrained assertion tests."""
    monkeypatch.setenv("BU_LAZY_MODELS", "0")

@pytest.fixture
def classifier_with_eager_loading(mocker, disable_lazy_loading):
    """Classifier with lazy loading disabled and mock references attached."""
    # ... implementation
```

## Migration Guide

### Before (Failing Test)
```python
def test_classifier_initialization(self, mocker):
    classifier = RealMLClassifier()
    mock_tokenizer.from_pretrained.assert_called_once()  # ❌ May fail with lazy loading
```

### After (Fixed Test)
```python
def test_classifier_initialization(self, mocker, disable_lazy_loading):
    classifier = RealMLClassifier()
    mock_tokenizer_patch.assert_called_once()  # ✅ Works reliably
```

The key changes:
1. Add `disable_lazy_loading` parameter to test function
2. Use the patch object reference instead of the mock object
3. Update assertions to use `mock_tokenizer_patch` instead of `mock_tokenizer.from_pretrained`

## Benefits

- ✅ **Reliable Assertions**: Tests work consistently regardless of lazy loading settings
- ✅ **Clear Intent**: Fixtures clearly indicate when eager loading is needed
- ✅ **Flexible**: Multiple approaches for different use cases
- ✅ **Backward Compatible**: Existing tests continue to work
- ✅ **Performance**: Default lazy loading still improves test performance for tests that don't need it
