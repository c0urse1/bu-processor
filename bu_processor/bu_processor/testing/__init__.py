"""
Testing utilities for bu_processor.

This module provides mock objects and test utilities to avoid common
mocking errors in PyTorch and transformer-based testing.
"""

from .mocks import (
    FakeTensor,
    FakeTokenizerOutput,
    FakeModelOutput,
    FakeTokenizer,
    FakeModel,
    FakeSentenceTransformer,
    create_mock_tokenizer,
    create_mock_model,
    create_mock_sentence_transformer,
    fake_tokenizer,  # legacy support
)

__all__ = [
    "FakeTensor",
    "FakeTokenizerOutput", 
    "FakeModelOutput",
    "FakeTokenizer",
    "FakeModel",
    "FakeSentenceTransformer",
    "create_mock_tokenizer",
    "create_mock_model",
    "create_mock_sentence_transformer",
    "fake_tokenizer",
]
