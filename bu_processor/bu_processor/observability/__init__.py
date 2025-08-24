# bu_processor/observability/__init__.py
"""
Observability package for metrics, tracing, and monitoring.

This package provides conditional imports based on feature flags:
- Metrics: Prometheus-compatible metrics with No-Op fallbacks
- Tracing: Distributed tracing (when enabled)
- Monitoring: Health checks and status monitoring
"""

from .metrics import (
    CounterClass,
    HistogramClass, 
    GaugeClass,
    ingest_errors,
    upsert_latency,
    document_count,
    create_counter,
    create_histogram,
    create_gauge
)

__all__ = [
    # Metric classes
    "CounterClass",
    "HistogramClass", 
    "GaugeClass",
    
    # Pre-defined metrics
    "ingest_errors",
    "upsert_latency", 
    "document_count",
    
    # Factory functions
    "create_counter",
    "create_histogram",
    "create_gauge"
]
