"""
Mock utilities for testing PyTorch and transformer components.

This module provides mock classes that properly implement the interfaces
expected by PyTorch tensors, transformers tokenizers, and models to avoid
AttributeError issues in tests.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union
import numpy as np


class FakeTensor:
    """
    Mock PyTorch tensor that implements common tensor methods.
    
    Prevents AttributeError: 'dict' object has no attribute 'to' errors
    when mocking PyTorch tensors in tests.
    """
    
    def __init__(self, arr: Union[np.ndarray, List, int, float]):
        if isinstance(arr, (int, float)):
            self.arr = np.array([arr])
        elif isinstance(arr, list):
            self.arr = np.array(arr)
        else:
            self.arr = arr
        self.shape = self.arr.shape
        self.dtype = self.arr.dtype
    
    def to(self, *args, **kwargs):
        """Mock PyTorch tensor .to() method for device/dtype conversion."""
        return self
    
    def cpu(self):
        """Mock PyTorch tensor .cpu() method."""
        return self
    
    def numpy(self):
        """Mock PyTorch tensor .numpy() method."""
        return self.arr
    
    def tolist(self):
        """Mock PyTorch tensor .tolist() method."""
        return self.arr.tolist()
    
    def item(self):
        """Mock PyTorch tensor .item() method for scalar tensors."""
        if self.arr.size == 1:
            return self.arr.item()
        raise ValueError("only one element tensors can be converted to Python scalars")
    
    def size(self, dim=None):
        """Mock PyTorch tensor .size() method."""
        if dim is None:
            return self.shape
        return self.shape[dim]
    
    def dim(self):
        """Mock PyTorch tensor .dim() method."""
        return len(self.shape)
    
    def __getitem__(self, key):
        """Support indexing like a real tensor."""
        result = self.arr[key]
        return FakeTensor(result)
    
    def __len__(self):
        """Support len() calls."""
        return len(self.arr)


class FakeTokenizerOutput:
    """
    Mock tokenizer output that implements the interface expected by transformers.
    
    Prevents errors when mocking tokenizer outputs in tests.
    """
    
    def __init__(self, input_ids: np.ndarray, attention_mask: Optional[np.ndarray] = None, 
                 token_type_ids: Optional[np.ndarray] = None):
        self.input_ids = FakeTensor(input_ids)
        self.attention_mask = FakeTensor(attention_mask if attention_mask is not None 
                                       else np.ones_like(input_ids))
        if token_type_ids is not None:
            self.token_type_ids = FakeTensor(token_type_ids)
    
    def to(self, *args, **kwargs):
        """Mock .to() method for device conversion."""
        return self
    
    def __getitem__(self, key):
        """Support dict-like access."""
        return getattr(self, key)
    
    def __contains__(self, key):
        """Support 'in' operator."""
        return hasattr(self, key)
    
    def keys(self):
        """Support .keys() method."""
        return ['input_ids', 'attention_mask', 'token_type_ids']


class FakeModelOutput:
    """
    Mock model output that implements the interface expected by transformers models.
    
    Prevents errors when mocking model outputs in tests.
    """
    
    def __init__(self, logits: np.ndarray, hidden_states: Optional[np.ndarray] = None,
                 attentions: Optional[np.ndarray] = None):
        self.logits = FakeTensor(logits)
        if hidden_states is not None:
            self.hidden_states = FakeTensor(hidden_states)
        if attentions is not None:
            self.attentions = FakeTensor(attentions)
    
    def __getitem__(self, key):
        """Support dict-like access."""
        return getattr(self, key)
    
    def __contains__(self, key):
        """Support 'in' operator."""
        return hasattr(self, key)


class FakeTokenizer:
    """
    Mock tokenizer that implements common transformers tokenizer methods.
    
    Provides a realistic interface for testing without requiring actual models.
    """
    
    def __init__(self, vocab_size: int = 1000, max_length: int = 512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.cls_token_id = 101
        self.unk_token_id = 100
    
    def __call__(self, texts: Union[str, List[str]], **kwargs) -> FakeTokenizerOutput:
        """Mock tokenization call."""
        return self.encode_plus(texts, **kwargs)
    
    def encode_plus(self, texts: Union[str, List[str]], **kwargs) -> FakeTokenizerOutput:
        """Mock encode_plus method."""
        if isinstance(texts, str):
            texts = [texts]
        
        max_length = kwargs.get('max_length', self.max_length)
        padding = kwargs.get('padding', False)
        truncation = kwargs.get('truncation', False)
        
        # Generate fake token IDs based on text length
        input_ids_list = []
        attention_mask_list = []
        
        for text in texts:
            # Simple simulation: 1 token per 4 characters + special tokens
            text_tokens = min(len(text) // 4 + 1, max_length - 2)
            seq_len = text_tokens + 2  # +2 for CLS and SEP
            
            if truncation and seq_len > max_length:
                seq_len = max_length
            
            # Generate token IDs
            token_ids = [self.cls_token_id] + list(range(1, seq_len - 1)) + [self.sep_token_id]
            attention_mask = [1] * len(token_ids)
            
            # Apply padding if requested
            if padding and len(token_ids) < max_length:
                pad_length = max_length - len(token_ids)
                token_ids.extend([self.pad_token_id] * pad_length)
                attention_mask.extend([0] * pad_length)
            
            input_ids_list.append(token_ids)
            attention_mask_list.append(attention_mask)
        
        # Convert to numpy arrays
        input_ids = np.array(input_ids_list, dtype=np.int64)
        attention_mask = np.array(attention_mask_list, dtype=np.int64)
        
        return FakeTokenizerOutput(input_ids, attention_mask)
    
    def decode(self, token_ids: Union[List[int], np.ndarray], **kwargs) -> str:
        """Mock decode method."""
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        return f"decoded_text_from_{len(token_ids)}_tokens"
    
    def batch_decode(self, token_ids_batch: Union[List[List[int]], np.ndarray], **kwargs) -> List[str]:
        """Mock batch_decode method."""
        return [self.decode(token_ids, **kwargs) for token_ids in token_ids_batch]


class FakeModel:
    """
    Mock transformer model that implements common model methods.
    
    Provides a realistic interface for testing without requiring actual models.
    """
    
    def __init__(self, num_labels: int = 2, hidden_size: int = 768):
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.config = SimpleNamespace(
            num_labels=num_labels,
            hidden_size=hidden_size,
            id2label={i: f"label_{i}" for i in range(num_labels)},
            label2id={f"label_{i}": i for i in range(num_labels)}
        )
        self.device = "cpu"
    
    def __call__(self, input_ids: FakeTensor, attention_mask: Optional[FakeTensor] = None, 
                 **kwargs) -> FakeModelOutput:
        """Mock forward pass."""
        batch_size = input_ids.size(0) if hasattr(input_ids, 'size') else input_ids.shape[0]
        
        # Generate fake logits
        logits = np.random.randn(batch_size, self.num_labels).astype(np.float32)
        
        # Make logits deterministic for testing
        np.random.seed(42)
        logits = np.random.randn(batch_size, self.num_labels).astype(np.float32)
        
        return FakeModelOutput(logits)
    
    def to(self, device):
        """Mock device movement."""
        self.device = device
        return self
    
    def eval(self):
        """Mock eval mode."""
        return self
    
    def train(self, mode=True):
        """Mock train mode."""
        return self


class FakeSentenceTransformer:
    """
    Mock SentenceTransformer that implements the encode method.
    
    Provides deterministic embeddings for testing semantic functionality.
    """
    
    def __init__(self, model_name: str = "fake-model", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or "cpu"
        self.dim = 384  # Common embedding dimension
    
    def encode(self, sentences: Union[str, List[str]], batch_size: int = 32, 
               convert_to_numpy: bool = True, normalize_embeddings: bool = False,
               **kwargs) -> np.ndarray:
        """Mock encode method with deterministic embeddings."""
        if isinstance(sentences, str):
            sentences = [sentences]
        
        embeddings = []
        for sentence in sentences:
            # Create deterministic embedding based on sentence hash
            import hashlib
            sentence_hash = hashlib.sha256(sentence.encode()).hexdigest()
            seed = int(sentence_hash[:8], 16) % (2**31)
            
            np.random.seed(seed)
            embedding = np.random.randn(self.dim).astype(np.float32)
            
            # Add some semantic structure for testing
            if "cat" in sentence.lower():
                embedding[0] += 2.0
            if "dog" in sentence.lower():
                embedding[1] += 2.0
            if "finance" in sentence.lower():
                embedding[2] += 2.0
            if "health" in sentence.lower():
                embedding[3] += 2.0
            
            if normalize_embeddings:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            embeddings.append(embedding)
        
        return np.stack(embeddings)


def create_mock_tokenizer(vocab_size: int = 1000, max_length: int = 512) -> FakeTokenizer:
    """Factory function to create a mock tokenizer."""
    return FakeTokenizer(vocab_size=vocab_size, max_length=max_length)


def create_mock_model(num_labels: int = 2, hidden_size: int = 768) -> FakeModel:
    """Factory function to create a mock model."""
    return FakeModel(num_labels=num_labels, hidden_size=hidden_size)


def create_mock_sentence_transformer(model_name: str = "fake-model", 
                                   device: Optional[str] = None) -> FakeSentenceTransformer:
    """Factory function to create a mock sentence transformer."""
    return FakeSentenceTransformer(model_name=model_name, device=device)


# Legacy support for existing tests
def fake_tokenizer(texts, **kwargs):
    """Legacy function for backward compatibility."""
    tokenizer = FakeTokenizer()
    return tokenizer(texts, **kwargs)
