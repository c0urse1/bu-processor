"""
Debug script to test imports and show the tracing functionality.
"""
import os
import sys

# Ensure we can use fake embeddings for testing
os.environ['EMBEDDINGS_BACKEND'] = 'fake'

try:
    from bu_processor.telemetry.trace import Trace, TraceLogger
    from bu_processor.factories import make_hybrid_retriever, make_answerer
    from bu_processor.answering.context_packer import pack_context
    from bu_processor.answering.synthesize import synthesize_answer
    
    print("✅ All imports successful!")
    
    # Test tracing
    tracer = Trace()
    tlog = TraceLogger(file_path=None)
    
    print("🔍 Testing trace functionality...")
    
    tracer.event("test_start", message="Starting debug test")
    
    with tracer.stage("test_stage"):
        import time
        time.sleep(0.1)  # Simulate work
    
    tracer.event("test_end", message="Test completed")
    
    print("📊 Trace data:")
    print(f"  - Trace ID: {tracer.trace_id}")
    print(f"  - Events: {len(tracer.events)}")
    
    for event in tracer.events:
        print(f"  - {event.name}: {event.payload}")
    
    tlog.log(tracer, extra={"debug": True})
    print("✅ Tracing test successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
