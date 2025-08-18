from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json, logging, time, uuid
from contextlib import contextmanager

@dataclass
class TraceEvent:
    t: float
    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    dur_ms: Optional[int] = None

@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: List[TraceEvent] = field(default_factory=list)

    def event(self, name: str, **payload: Any) -> None:
        self.events.append(TraceEvent(t=time.time(), name=name, payload=payload))

    @contextmanager
    def stage(self, name: str, **payload: Any):
        start = time.perf_counter()
        self.event(f"{name}.start", **payload)
        try:
            yield
        finally:
            dur = int(1000 * (time.perf_counter() - start))
            self.events.append(TraceEvent(t=time.time(), name=f"{name}.end", payload=payload, dur_ms=dur))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "events": [asdict(e) for e in self.events],
        }

class TraceLogger:
    """
    Writes JSON lines to std logging and (optionally) a file.
    """
    def __init__(self, logger_name: str = "bu.trace", file_path: Optional[str] = None):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        if file_path:
            fh = logging.FileHandler(file_path, encoding="utf-8")
            fh.setLevel(logging.INFO)
            self.logger.addHandler(fh)

    def log(self, trace: Trace, extra: Optional[Dict[str, Any]] = None):
        obj = trace.to_dict()
        if extra:
            obj.update(extra)
        self.logger.info(json.dumps(obj, ensure_ascii=False))
