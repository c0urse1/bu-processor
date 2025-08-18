import json
import tempfile
from bu_processor.telemetry.trace import Trace, TraceLogger, TraceEvent
from bu_processor.telemetry.wrap import serialize_hits
from bu_processor.retrieval.models import RetrievalHit

def test_trace_basic():
    trace = Trace()
    trace.event("test", key="value")
    
    with trace.stage("process"):
        pass  # simulated work
    
    data = trace.to_dict()
    assert "trace_id" in data
    assert len(data["events"]) >= 3  # test, process.start, process.end
    assert data["events"][0]["name"] == "test"
    assert data["events"][1]["name"] == "process.start"
    assert data["events"][2]["name"] == "process.end"
    assert data["events"][2]["dur_ms"] is not None

def test_trace_logger():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        logger = TraceLogger(file_path=f.name)
        trace = Trace()
        trace.event("test_event", data="test")
        logger.log(trace, extra={"query": "test query"})
        
    # Verify file was written
    with open(f.name, 'r') as f:
        line = f.read().strip()
        data = json.loads(line)
        assert "trace_id" in data
        assert "query" in data
        assert data["query"] == "test query"

def test_serialize_hits():
    hits = [
        RetrievalHit(
            id="chunk_1",
            text="Test text",
            score=0.95,
            metadata={"doc_id": "doc1", "section": "intro", "page": 1}
        )
    ]
    
    serialized = serialize_hits(hits)
    assert len(serialized) == 1
    assert serialized[0]["id"] == "chunk_1"
    assert serialized[0]["score"] == 0.95
    assert serialized[0]["meta"]["doc_id"] == "doc1"
    assert serialized[0]["meta"]["section"] == "intro"
    assert serialized[0]["meta"]["page"] == 1
