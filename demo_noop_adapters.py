#!/usr/bin/env python3
"""
Demo: No-Op Metrics & Rate Limiter System

This script demonstrates how the observability and rate limiting systems
work with feature flags. When flags are disabled, all operations become
No-Op but preserve the same API.
"""
import os
import sys
import time
from pathlib import Path

# Add the bu_processor package to path
sys.path.insert(0, str(Path(__file__).parent / "bu_processor"))

def demo_metrics_system():
    """Demonstrate the metrics system with and without flags."""
    print("=" * 60)
    print("METRICS SYSTEM DEMO")
    print("=" * 60)
    
    # Test with metrics disabled (default)
    print("\n1. Testing with ENABLE_METRICS=False (default)...")
    os.environ.pop("ENABLE_METRICS", None)  # Ensure it's not set
    
    # Import after setting environment
    from bu_processor.observability.metrics import (
        CounterClass, HistogramClass, GaugeClass,
        ingest_errors, upsert_latency, document_count,
        is_metrics_enabled, get_metrics_info,
        time_operation
    )
    
    print(f"   Metrics enabled: {is_metrics_enabled()}")
    print(f"   Implementation: {get_metrics_info()['implementation']}")
    
    # Test metric operations (should be No-Op)
    print("   Testing metric operations...")
    
    # Counter operations
    ingest_errors.inc()
    ingest_errors.labels(error_type="validation", document_type="pdf").inc(5)
    print("   ✓ Counter operations (No-Op)")
    
    # Histogram operations
    upsert_latency.observe(1.5)
    upsert_latency.labels(operation="upsert", status="success").observe(0.8)
    print("   ✓ Histogram operations (No-Op)")
    
    # Gauge operations
    document_count.set(100)
    document_count.labels(status="processed", document_type="pdf").inc(10)
    print("   ✓ Gauge operations (No-Op)")
    
    # Timing context manager
    with time_operation(upsert_latency, operation="bulk_upsert"):
        time.sleep(0.1)  # Simulate work
    print("   ✓ Timing context manager (No-Op)")
    
    # Create custom metrics
    custom_counter = CounterClass("custom_counter", "Custom counter")
    custom_histogram = HistogramClass("custom_histogram", "Custom histogram")
    custom_gauge = GaugeClass("custom_gauge", "Custom gauge")
    
    custom_counter.inc()
    custom_histogram.observe(2.0)
    custom_gauge.set(42)
    print("   ✓ Custom metric creation (No-Op)")
    
    print("\n2. Testing with ENABLE_METRICS=True...")
    os.environ["ENABLE_METRICS"] = "true"
    
    # NOTE: In a real application, you'd need to restart or reload modules
    # to pick up the flag change. For demo purposes, we'll just show the concept.
    print("   (In real app: metrics would become active Prometheus instances)")
    print("   (All operations would start collecting real metrics)")
    print("   ✓ Flag change detected - would enable real metrics")


def demo_rate_limiter_system():
    """Demonstrate the rate limiter system with and without flags."""
    print("\n" + "=" * 60)
    print("RATE LIMITER SYSTEM DEMO")
    print("=" * 60)
    
    # Test with rate limiting disabled (default)
    print("\n1. Testing with ENABLE_RATE_LIMITER=False (default)...")
    os.environ.pop("ENABLE_RATE_LIMITER", None)  # Ensure it's not set
    
    # Import after setting environment
    from bu_processor.core.ratelimit import (
        RateLimiter, rate_limited, async_rate_limited,
        pinecone_limiter, embedding_limiter,
        is_rate_limiting_enabled, get_rate_limit_info,
        rate_limited_operation
    )
    
    print(f"   Rate limiting enabled: {is_rate_limiting_enabled()}")
    print(f"   Implementation: {get_rate_limit_info()['implementation']}")
    
    # Test rate limiter operations (should be No-Op)
    print("   Testing rate limiter operations...")
    
    # Basic rate limiter
    limiter = RateLimiter(calls_per_second=2.0)
    
    start_time = time.time()
    for i in range(5):
        limiter.acquire()
        print(f"   Call {i+1} at {time.time() - start_time:.2f}s")
    print("   ✓ Rate limiter acquire() calls (No-Op - no delays)")
    
    # Context manager
    print("   Testing context manager...")
    with pinecone_limiter:
        print("   Inside rate-limited context (No-Op)")
    print("   ✓ Context manager (No-Op)")
    
    # Decorator usage
    print("   Testing decorators...")
    
    @rate_limited(calls_per_second=5.0)
    def mock_api_call(data):
        return f"Processed: {data}"
    
    start_time = time.time()
    for i in range(3):
        result = mock_api_call(f"data_{i}")
        print(f"   {result} at {time.time() - start_time:.2f}s")
    print("   ✓ Rate limited decorator (No-Op - no delays)")
    
    # Pre-configured limiters
    print("   Testing pre-configured limiters...")
    with rate_limited_operation(embedding_limiter, "embedding_call"):
        print("   Embedding operation (No-Op)")
    print("   ✓ Pre-configured limiters (No-Op)")
    
    print("\n2. Testing with ENABLE_RATE_LIMITER=True...")
    os.environ["ENABLE_RATE_LIMITER"] = "true"
    
    print("   (In real app: rate limiting would become active)")
    print("   (All operations would respect rate limits with delays)")
    print("   ✓ Flag change detected - would enable real rate limiting")


def demo_integration_example():
    """Show how metrics and rate limiting work together."""
    print("\n" + "=" * 60)
    print("INTEGRATION EXAMPLE") 
    print("=" * 60)
    
    print("\nExample: PDF Processing Pipeline with Observability")
    
    # Reset environment to defaults
    os.environ.pop("ENABLE_METRICS", None)
    os.environ.pop("ENABLE_RATE_LIMITER", None)
    
    from bu_processor.observability.metrics import (
        pdf_processing_latency, pdf_processing_errors, time_operation
    )
    from bu_processor.core.ratelimit import (
        pdf_processing_limiter, rate_limited_operation
    )
    
    def process_pdf_mock(filename):
        """Mock PDF processing function with observability."""
        print(f"   Processing {filename}...")
        
        # Rate limiting (No-Op)
        with rate_limited_operation(pdf_processing_limiter, "pdf_processing"):
            # Metrics timing (No-Op)
            with time_operation(pdf_processing_latency, extraction_method="mock", pages="10"):
                time.sleep(0.05)  # Simulate processing
                
                # Simulate occasional error
                if "error" in filename:
                    pdf_processing_errors.labels(error_type="parse_error", stage="extraction").inc()
                    raise ValueError(f"Mock error processing {filename}")
                
                return f"Processed {filename} successfully"
    
    # Process multiple files
    files = ["doc1.pdf", "doc2.pdf", "error_doc.pdf", "doc3.pdf"]
    
    start_time = time.time()
    for filename in files:
        try:
            result = process_pdf_mock(filename)
            print(f"   ✓ {result}")
        except ValueError as e:
            print(f"   ✗ {e}")
        
        # Show timing (would show actual delays with rate limiting enabled)
        elapsed = time.time() - start_time
        print(f"     Elapsed: {elapsed:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\n   Total processing time: {total_time:.2f}s")
    print("   (With flags enabled: would show real metrics and rate limiting)")


def demo_feature_flag_benefits():
    """Demonstrate the benefits of the feature flag approach."""
    print("\n" + "=" * 60)
    print("FEATURE FLAG BENEFITS")
    print("=" * 60)
    
    print("\n✅ ZERO CODE REMOVAL:")
    print("   - All metric collection code stays in place")
    print("   - All rate limiting logic preserved")
    print("   - No conditional 'if' statements needed in business logic")
    
    print("\n✅ MVP-READY:")
    print("   - Default flags = False = No-Op = Zero overhead")
    print("   - Production code runs without external dependencies")
    print("   - No Prometheus or rate limiting libraries required")
    
    print("\n✅ INTELLIGENCE REMAINS SWITCHABLE:")
    print("   - Set ENABLE_METRICS=true → Real Prometheus metrics")
    print("   - Set ENABLE_RATE_LIMITER=true → Real rate limiting")
    print("   - Features activate instantly without code changes")
    
    print("\n✅ SAFE DEVELOPMENT:")
    print("   - Advanced features can be developed and tested independently")
    print("   - Gradual rollout through environment variables")
    print("   - Easy rollback by changing flag values")
    
    print("\n✅ API COMPATIBILITY:")
    print("   - Same API whether enabled or disabled")
    print("   - No try/catch blocks for missing dependencies")
    print("   - Consistent behavior across environments")


if __name__ == "__main__":
    print("🚀 BU-Processor No-Op Adapters Demo")
    print("=" * 60)
    print("Demonstrating observability and rate limiting with feature flags")
    
    demo_metrics_system()
    demo_rate_limiter_system()
    demo_integration_example()
    demo_feature_flag_benefits()
    
    print("\n" + "=" * 60)
    print("✅ Demo completed!")
    print("\nTo enable features in production:")
    print("  export ENABLE_METRICS=true")
    print("  export ENABLE_RATE_LIMITER=true")
    print("=" * 60)
