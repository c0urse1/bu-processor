# Standardized Wiring Implementation Summary

## 🎯 Overview

This implementation provides a **standardized factory function** (`make_pinecone_manager`) that ensures consistent wiring across all CLI/Worker/API components. This eliminates configuration drift and provides a single source of truth for Pinecone manager creation.

## 🔌 Standardized Wiring Pattern

### Required Import Pattern
```python
from bu_processor.integrations.pinecone_facade import make_pinecone_manager
from bu_processor.embeddings.embedder import Embedder
```

### Standard Initialization Pattern
```python
embedder = Embedder()
pc = make_pinecone_manager(
    index_name=os.getenv("PINECONE_INDEX_NAME", "bu-processor"),
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV"),  # v2
    cloud=os.getenv("PINECONE_CLOUD"),      # v3
    region=os.getenv("PINECONE_REGION"),    # v3
    namespace=os.getenv("PINECONE_NAMESPACE")
)
pc.ensure_index(embedder.dimension)
```

## 🏗️ Implementation Details

### Factory Function Signature
```python
def make_pinecone_manager(
    index_name: str,
    api_key: Optional[str] = None,
    environment: Optional[str] = None,  # v2
    cloud: Optional[str] = None,        # v3 serverless
    region: Optional[str] = None,       # v3 serverless
    metric: str = "cosine",
    namespace: Optional[str] = None,
    force_simple: bool = False
) -> PineconeManager
```

### Environment Variable Defaults
The factory function automatically uses environment variables as defaults:
- `PINECONE_API_KEY` → `api_key`
- `PINECONE_ENV` → `environment` (v2)
- `PINECONE_CLOUD` → `cloud` (v3)
- `PINECONE_REGION` → `region` (v3)
- `PINECONE_NAMESPACE` → `namespace`

### Facade Integration
- Returns `PineconeManager` facade instance
- Automatically selects simple vs enhanced implementation
- Inherits all quality gates and reranking capabilities
- Maintains unified upsert signature across implementations

## 📁 Updated Components

### 1. CLI Components
**`bu_processor/cli_ingest.py`**
```python
# Before
from bu_processor.integrations.pinecone_manager import PineconeManager

# After  
from bu_processor.integrations.pinecone_facade import make_pinecone_manager
```

### 2. Factory Functions
**`bu_processor/factories.py`**
```python
def make_simplified_pinecone_manager():
    """Create the new simplified Pinecone manager using standardized wiring."""
    from bu_processor.integrations.pinecone_facade import make_pinecone_manager
    import os
    
    return make_pinecone_manager(
        index_name=os.getenv("PINECONE_INDEX_NAME", "bu-processor"),
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV"),       # v2
        cloud=os.getenv("PINECONE_CLOUD"),           # v3
        region=os.getenv("PINECONE_REGION"),         # v3
        namespace=os.getenv("PINECONE_NAMESPACE")
    )
```

### 3. Pipeline Components
**`bu_processor/pipeline/simplified_upsert.py`**
```python
# Before
from ..integrations.pinecone_manager import PineconeManager

# After
from ..integrations.pinecone_facade import make_pinecone_manager

# Initialization
self.pinecone_manager = pinecone_manager or make_pinecone_manager(
    index_name=os.getenv("PINECONE_INDEX_NAME", "bu-processor"),
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV"),   # v2
    cloud=os.getenv("PINECONE_CLOUD"),       # v3
    region=os.getenv("PINECONE_REGION"),     # v3
    namespace=self.namespace
)
```

### 4. Background Jobs
**`bu_processor/ingest.py`**
```python
# Before
from .integrations.pinecone_manager import PineconeManager

# After
from .integrations.pinecone_facade import make_pinecone_manager

# Initialization
self.pinecone_manager = make_pinecone_manager(
    index_name=self.config.vector_db.pinecone_index_name,
    api_key=self.config.vector_db.pinecone_api_key,
    environment=self.config.vector_db.pinecone_env,   # v2
    cloud=self.config.vector_db.pinecone_cloud,       # v3
    region=self.config.vector_db.pinecone_region,     # v3
    namespace=self.config.vector_db.pinecone_namespace
)
```

### 5. Demo Scripts
**`scripts/demo_wiring.py`**
```python
# Before
from bu_processor.integrations.pinecone_manager import PineconeManager

# After
from bu_processor.integrations.pinecone_facade import make_pinecone_manager
```

## ⚙️ Configuration Management

### Environment Variables
```bash
# Required
export PINECONE_API_KEY="your-api-key"
export PINECONE_INDEX_NAME="bu-processor"

# Version-specific (choose one)
# v2 Configuration
export PINECONE_ENV="your-environment"

# v3 Serverless Configuration  
export PINECONE_CLOUD="gcp"         # or "aws"
export PINECONE_REGION="us-west1"   # region code

# Optional
export PINECONE_NAMESPACE="your-namespace"
```

### Type Safety
```python
# Import with TYPE_CHECKING for proper type annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..integrations.pinecone_facade import PineconeManager

# Use quoted type annotations
def func(manager: Optional["PineconeManager"] = None):
    pass
```

## 🎯 Benefits

### 1. Consistency
- **Single Source of Truth**: All components use the same factory function
- **Unified Configuration**: Environment variables handled consistently
- **Version Compatibility**: Supports both v2 and v3 automatically

### 2. Maintainability
- **Centralized Changes**: Update factory function affects all components
- **Environment Defaults**: Reduces boilerplate configuration code
- **Type Safety**: Proper type annotations with TYPE_CHECKING

### 3. Features
- **Quality Gates**: Automatic dimension checks and data validation
- **Reranking**: Optional cross-encoder intelligence booster
- **Facade Pattern**: Automatic simple vs enhanced selection
- **Observability**: Built-in metrics and rate limiting

### 4. Deployment
- **Easy Configuration**: Single set of environment variables
- **Flexible Overrides**: Can override any parameter when needed
- **Testing Support**: `force_simple=True` for controlled testing

## 📋 Migration Checklist

- ✅ **Factory Function**: `make_pinecone_manager` implemented in facade
- ✅ **CLI Components**: Updated `cli_ingest.py` imports
- ✅ **Factory Module**: Updated `factories.py` to use standardized wiring
- ✅ **Pipeline Components**: Updated `simplified_upsert.py` imports and initialization
- ✅ **Background Jobs**: Updated `ingest.py` imports and initialization  
- ✅ **Demo Scripts**: Updated `demo_wiring.py` imports
- ✅ **Type Safety**: Added TYPE_CHECKING imports where needed

## 🚀 Usage Examples

### CLI Component
```python
#!/usr/bin/env python3
import os
from bu_processor.integrations.pinecone_facade import make_pinecone_manager
from bu_processor.embeddings.embedder import Embedder

def main():
    embedder = Embedder()
    pc = make_pinecone_manager(
        index_name=os.getenv("PINECONE_INDEX_NAME", "bu-processor"),
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV"),  # v2
        cloud=os.getenv("PINECONE_CLOUD"),      # v3
        region=os.getenv("PINECONE_REGION"),    # v3
        namespace=os.getenv("PINECONE_NAMESPACE")
    )
    pc.ensure_index(embedder.dimension)
    
    # Use pc for operations...
```

### API Endpoint
```python
# Global initialization
embedder = Embedder()
pc = make_pinecone_manager(
    index_name=os.getenv("PINECONE_INDEX_NAME", "bu-processor"),
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV"),  # v2
    cloud=os.getenv("PINECONE_CLOUD"),      # v3
    region=os.getenv("PINECONE_REGION"),    # v3
    namespace=os.getenv("PINECONE_NAMESPACE")
)
pc.ensure_index(embedder.dimension)

@app.route("/search")
def search():
    results = pc.query_by_text(
        text=request.args.get("q"),
        embedder=embedder,
        enable_rerank=True  # Use quality intelligence booster
    )
    return jsonify(results)
```

### Worker Process
```python
class DocumentWorker:
    def __init__(self):
        self.embedder = Embedder()
        self.pc = make_pinecone_manager(
            index_name=os.getenv("PINECONE_INDEX_NAME", "bu-processor"),
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV"),  # v2
            cloud=os.getenv("PINECONE_CLOUD"),      # v3
            region=os.getenv("PINECONE_REGION"),    # v3
            namespace=os.getenv("PINECONE_NAMESPACE")
        )
        self.pc.ensure_index(self.embedder.dimension)
    
    def process_document(self, doc):
        # Use self.pc with quality gates
        self.pc.upsert_vectors(
            ids=doc_ids,
            vectors=doc_vectors,
            metadatas=doc_metadata,
            embedder=self.embedder  # Triggers quality gates
        )
```

This standardized wiring approach ensures consistent, maintainable, and feature-rich Pinecone integration across all components of the BU-Processor system.
