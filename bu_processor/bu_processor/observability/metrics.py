# bu_processor/observability/metrics.py
"""
Metrics collection with feature flag control.

This module provides Prometheus-compatible metrics that can be disabled
via feature flags. When ENABLE_METRICS=False, all metrics become No-Op
classes that do nothing but preserve the API.

Usage:
    from bu_processor.observability.metrics import CounterClass, HistogramClass
    
    # These will be real Prometheus metrics if enabled, No-Op otherwise
    errors = CounterClass("errors_total", "Total errors")
    latency = HistogramClass("latency_seconds", "Operation latency")
    
    errors.inc()  # Works regardless of flag state
    latency.observe(0.5)
"""
from typing import Any, Optional, List, Union
from ..core.flags import ENABLE_METRICS

# ============================================================================
# No-Op Metric Classes (used when ENABLE_METRICS=False)
# ============================================================================

class _NoOpCounter:
    """No-Op counter that does nothing but preserves the API."""
    
    def __init__(self, name: str, documentation: str, labelnames: Optional[List[str]] = None):
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames or []
    
    def inc(self, amount: Union[int, float] = 1, **labels) -> None:
        """Increment counter (No-Op)."""
        pass
    
    def labels(self, **labels) -> '_NoOpCounter':
        """Return labeled version (No-Op)."""
        return self
    
    def __call__(self, **labels) -> '_NoOpCounter':
        """Return labeled version (No-Op)."""
        return self


class _NoOpHistogram:
    """No-Op histogram that does nothing but preserves the API."""
    
    def __init__(self, name: str, documentation: str, 
                 labelnames: Optional[List[str]] = None,
                 buckets: Optional[List[float]] = None):
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames or []
        self.buckets = buckets
    
    def observe(self, amount: Union[int, float], **labels) -> None:
        """Observe value (No-Op)."""
        pass
    
    def time(self):
        """Context manager for timing (No-Op)."""
        return _NoOpTimer()
    
    def labels(self, **labels) -> '_NoOpHistogram':
        """Return labeled version (No-Op)."""
        return self
    
    def __call__(self, **labels) -> '_NoOpHistogram':
        """Return labeled version (No-Op)."""
        return self


class _NoOpGauge:
    """No-Op gauge that does nothing but preserves the API."""
    
    def __init__(self, name: str, documentation: str, labelnames: Optional[List[str]] = None):
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames or []
    
    def set(self, value: Union[int, float], **labels) -> None:
        """Set gauge value (No-Op)."""
        pass
    
    def inc(self, amount: Union[int, float] = 1, **labels) -> None:
        """Increment gauge (No-Op)."""
        pass
    
    def dec(self, amount: Union[int, float] = 1, **labels) -> None:
        """Decrement gauge (No-Op)."""
        pass
    
    def labels(self, **labels) -> '_NoOpGauge':
        """Return labeled version (No-Op)."""
        return self
    
    def __call__(self, **labels) -> '_NoOpGauge':
        """Return labeled version (No-Op)."""
        return self


class _NoOpTimer:
    """No-Op timer context manager."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# ============================================================================
# Conditional Import and Class Assignment
# ============================================================================

if ENABLE_METRICS:
    try:
        # Import real Prometheus classes
        from prometheus_client import Counter, Histogram, Gauge
        CounterClass = Counter
        HistogramClass = Histogram
        GaugeClass = Gauge
        _metrics_available = True
    except ImportError:
        # Fallback to No-Op if prometheus_client not available
        CounterClass = _NoOpCounter
        HistogramClass = _NoOpHistogram
        GaugeClass = _NoOpGauge
        _metrics_available = False
else:
    # Use No-Op classes when metrics disabled
    CounterClass = _NoOpCounter
    HistogramClass = _NoOpHistogram
    GaugeClass = _NoOpGauge
    _metrics_available = False


# ============================================================================
# Pre-defined Application Metrics
# ============================================================================

# Ingest & Processing Metrics
ingest_errors = CounterClass(
    "bu_processor_ingest_errors_total", 
    "Total number of document ingestion errors",
    labelnames=["error_type", "document_type"]
)

upsert_latency = HistogramClass(
    "bu_processor_upsert_seconds", 
    "Time spent upserting documents to vector database",
    labelnames=["operation", "status"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

document_count = GaugeClass(
    "bu_processor_documents_total",
    "Total number of processed documents",
    labelnames=["status", "document_type"]
)

# Embedding Metrics  
embedding_latency = HistogramClass(
    "bu_processor_embedding_seconds",
    "Time spent generating embeddings",
    labelnames=["model", "batch_size"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

embedding_cache_hits = CounterClass(
    "bu_processor_embedding_cache_hits_total",
    "Number of embedding cache hits",
    labelnames=["cache_type"]
)

embedding_cache_misses = CounterClass(
    "bu_processor_embedding_cache_misses_total", 
    "Number of embedding cache misses",
    labelnames=["cache_type"]
)

# PDF Processing Metrics
pdf_processing_latency = HistogramClass(
    "bu_processor_pdf_processing_seconds",
    "Time spent processing PDF documents",
    labelnames=["extraction_method", "pages"],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
)

pdf_processing_errors = CounterClass(
    "bu_processor_pdf_errors_total",
    "Number of PDF processing errors", 
    labelnames=["error_type", "stage"]
)

# Pinecone Integration Metrics
pinecone_operations = CounterClass(
    "bu_processor_pinecone_operations_total",
    "Total Pinecone operations performed",
    labelnames=["operation", "status"]
)

pinecone_latency = HistogramClass(
    "bu_processor_pinecone_seconds",
    "Pinecone operation latency",
    labelnames=["operation"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Classification Metrics
classification_accuracy = GaugeClass(
    "bu_processor_classification_accuracy",
    "Classification accuracy score",
    labelnames=["model", "dataset"]
)

classification_latency = HistogramClass(
    "bu_processor_classification_seconds", 
    "Time spent on classification",
    labelnames=["model", "batch_size"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
)


# ============================================================================
# Factory Functions
# ============================================================================

def create_counter(name: str, documentation: str, 
                  labelnames: Optional[List[str]] = None) -> Union[CounterClass, _NoOpCounter]:
    """Create a counter metric."""
    return CounterClass(name, documentation, labelnames)


def create_histogram(name: str, documentation: str,
                    labelnames: Optional[List[str]] = None,
                    buckets: Optional[List[float]] = None) -> Union[HistogramClass, _NoOpHistogram]:
    """Create a histogram metric."""
    return HistogramClass(name, documentation, labelnames, buckets)


def create_gauge(name: str, documentation: str,
                labelnames: Optional[List[str]] = None) -> Union[GaugeClass, _NoOpGauge]:
    """Create a gauge metric."""
    return GaugeClass(name, documentation, labelnames)


# ============================================================================
# Utility Functions
# ============================================================================

def is_metrics_enabled() -> bool:
    """Check if metrics collection is enabled."""
    return ENABLE_METRICS and _metrics_available


def get_metrics_info() -> dict:
    """Get information about metrics system status."""
    return {
        "enabled": ENABLE_METRICS,
        "available": _metrics_available,
        "implementation": "prometheus" if _metrics_available and ENABLE_METRICS else "noop",
        "total_metrics": len([
            ingest_errors, upsert_latency, document_count,
            embedding_latency, embedding_cache_hits, embedding_cache_misses,
            pdf_processing_latency, pdf_processing_errors,
            pinecone_operations, pinecone_latency,
            classification_accuracy, classification_latency
        ])
    }


# ============================================================================
# Context Managers for Timing
# ============================================================================

class MetricTimer:
    """Context manager for timing operations with metrics."""
    
    def __init__(self, histogram_metric, **labels):
        self.histogram = histogram_metric
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            import time
            duration = time.time() - self.start_time
            if self.labels:
                self.histogram.labels(**self.labels).observe(duration)
            else:
                self.histogram.observe(duration)


def time_operation(histogram_metric, **labels):
    """Context manager factory for timing operations."""
    return MetricTimer(histogram_metric, **labels)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Metric classes (conditional)
    "CounterClass",
    "HistogramClass", 
    "GaugeClass",
    
    # Pre-defined metrics
    "ingest_errors",
    "upsert_latency",
    "document_count", 
    "embedding_latency",
    "embedding_cache_hits",
    "embedding_cache_misses",
    "pdf_processing_latency",
    "pdf_processing_errors",
    "pinecone_operations",
    "pinecone_latency",
    "classification_accuracy",
    "classification_latency",
    
    # Factory functions
    "create_counter",
    "create_histogram",
    "create_gauge",
    
    # Utilities
    "is_metrics_enabled",
    "get_metrics_info",
    "time_operation",
    "MetricTimer",
    
    # No-Op classes (for direct use if needed)
    "_NoOpCounter",
    "_NoOpHistogram", 
    "_NoOpGauge"
]
