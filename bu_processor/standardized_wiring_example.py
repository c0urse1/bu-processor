#!/usr/bin/env python3
"""
Standardized Wiring Example
===========================

This demonstrates the recommended wiring pattern for all CLI/Worker/API components.
Use this pattern consistently across the entire system.
"""

import os

def standard_wiring_example():
    """
    Demonstrate the standardized wiring pattern.
    This is the recommended approach for all CLI/Worker/API components.
    """
    print('🔌 Standardized Wiring Example')
    print('=' * 50)
    
    # Step 1: Import the standardized factory functions
    from bu_processor.integrations.pinecone_facade import make_pinecone_manager
    from bu_processor.embeddings.embedder import Embedder
    
    print('✅ Imported standardized factory functions')
    
    # Step 2: Initialize embedder
    embedder = Embedder()
    print(f'✅ Embedder initialized (dimension: {embedder.dimension})')
    
    # Step 3: Create Pinecone manager using standardized wiring
    pc = make_pinecone_manager(
        index_name=os.getenv("PINECONE_INDEX_NAME", "bu-processor"),
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV"),  # v2
        cloud=os.getenv("PINECONE_CLOUD"),      # v3
        region=os.getenv("PINECONE_REGION"),    # v3
        namespace=os.getenv("PINECONE_NAMESPACE")
    )
    print(f'✅ Pinecone manager created via standardized wiring')
    print(f'   Implementation: {pc.implementation_type}')
    print(f'   Enhanced: {pc.is_enhanced}')
    
    # Step 4: Ensure index with correct dimension
    try:
        pc.ensure_index(embedder.dimension)
        print(f'✅ Index ensured with dimension: {embedder.dimension}')
    except Exception as e:
        print(f'⚠️  Index ensure failed (expected without credentials): {e}')
    
    print()
    print('🎯 Standardized Wiring Complete!')
    print()
    print('📋 This Pattern Should Be Used In:')
    print('   - CLI components (cli_ingest.py, etc.)')
    print('   - Worker processes')
    print('   - API endpoints')
    print('   - Pipeline factories')
    print('   - Integration tests')
    print()
    print('🔧 Benefits:')
    print('   ✅ Consistent initialization across all components')
    print('   ✅ Environment variable defaults built-in')
    print('   ✅ Automatic facade selection (simple vs enhanced)')
    print('   ✅ Quality gates and reranking ready')
    print('   ✅ Easy to maintain and debug')

def cli_example():
    """Example CLI wiring."""
    print('\n📱 CLI Example:')
    print('-' * 20)
    print('''
# cli_component.py
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
''')

def api_example():
    """Example API wiring."""
    print('\n🌐 API Example:')
    print('-' * 20)
    print('''
# api_endpoint.py
import os
from bu_processor.integrations.pinecone_facade import make_pinecone_manager
from bu_processor.embeddings.embedder import Embedder

# Initialize once at startup
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
def search_endpoint():
    # Use pc for search operations...
    pass
''')

def worker_example():
    """Example Worker wiring."""
    print('\n⚙️  Worker Example:')
    print('-' * 20)
    print('''
# worker_process.py
import os
from bu_processor.integrations.pinecone_facade import make_pinecone_manager
from bu_processor.embeddings.embedder import Embedder

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
        # Use self.pc for document processing...
        pass
''')

if __name__ == '__main__':
    standard_wiring_example()
    cli_example()
    api_example()
    worker_example()
    
    print('\n🚀 Ready for standardized deployment!')
