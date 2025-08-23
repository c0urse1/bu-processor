# 🚀 INGESTION JOB/WORKER SYSTEM DOCUMENTATION

## Übersicht

Das BU-Processor System bietet jetzt robuste, skalierbare PDF-Ingestion durch ein Job/Worker-System mit sofortiger 202-Antwort und Background-Verarbeitung.

## Architektur

```
Client Upload → API (202 Response) → Background Worker → Storage (Pinecone + SQLite)
     ↑                    ↓                    ↓
   Job ID              Job Status         Retry/Recovery
```

## API Endpoints

### 1. Background PDF Ingestion

```bash
POST /ingest/pdf
Content-Type: multipart/form-data
Authorization: Bearer <BU_API_TOKEN>

# Response (sofort 202)
{
  "status": "accepted",
  "job_id": "uuid-here",
  "filename": "document.pdf",
  "file_size": 1024000,
  "created_at": "2025-08-23T10:30:00Z",
  "tracking_url": "/ingest/status/uuid-here"
}
```

### 2. Job Status Tracking

```bash
GET /ingest/status/{job_id}
Authorization: Bearer <BU_API_TOKEN>

# Response
{
  "job_id": "uuid-here",
  "filename": "document.pdf", 
  "status": "completed",  # pending|running|completed|failed|retrying
  "created_at": "2025-08-23T10:30:00Z",
  "started_at": "2025-08-23T10:30:05Z",
  "completed_at": "2025-08-23T10:30:45Z",
  "processing_duration": 40.2,
  "retry_count": 0,
  "result": {
    "classification": {
      "predicted_label": "BU_Antrag",
      "confidence": 0.89
    },
    "storage": {
      "sqlite": {"status": "success", "document_id": "doc_123"},
      "pinecone": {"status": "success", "chunks_stored": 5}
    }
  }
}
```

### 3. Job Listing

```bash
GET /ingest/jobs
Authorization: Bearer <BU_API_TOKEN>

# Response
{
  "jobs": [...],
  "total": 25,
  "active": 3,
  "completed": 20,
  "failed": 2
}
```

## Zentrale process_pdf Funktion

Die neue `process_pdf()` Funktion wird sowohl von API als auch CLI verwendet:

```python
from bu_processor.ingest import process_pdf

# Für API und CLI verwendbar
result = process_pdf(
    file_path="document.pdf",
    output_dir="results/",      # Optional für CLI
    store_in_pinecone=True,     # Konfigurierbar
    store_in_sqlite=True        # Konfigurierbar
)
```

## CLI Interface

Das neue CLI-Tool nutzt dieselbe Logik wie die API:

```bash
# Einzelne Datei
python scripts/process_pdf_cli.py document.pdf

# Batch-Verarbeitung
python scripts/process_pdf_cli.py --batch data/pdfs/

# Mit Ausgabe-Verzeichnis
python scripts/process_pdf_cli.py document.pdf --output results/

# Nur Pinecone (ohne SQLite)
python scripts/process_pdf_cli.py document.pdf --pinecone-only

# Nur SQLite (ohne Pinecone)
python scripts/process_pdf_cli.py document.pdf --sqlite-only

# Verbose Ausgabe
python scripts/process_pdf_cli.py document.pdf --verbose
```

## Error Handling & Retry

- **Automatische Retries**: Bei temporären Fehlern (max 3 Versuche)
- **Exponential Backoff**: 2, 4, 8 Sekunden Wartezeit
- **Detaillierte Logs**: Strukturiertes Logging mit Kontext
- **Graceful Degradation**: Partial Success bei Storage-Fehlern

## Job Status Lifecycle

```
pending → running → completed ✅
   ↓         ↓          ↑
   ↓    (error) → retrying → (retry limit) → failed ❌
   ↓                     ↑
   → failed ❌           ↑
```

## Storage Strategy

### Dual Storage Approach
- **SQLite**: Vollständige Dokumente + Metadaten
- **Pinecone**: Chunked Embeddings für Similarity Search

### Metadata Enrichment
```json
{
  "filename": "antrag.pdf",
  "classification": {
    "predicted_label": "BU_Antrag",
    "predicted_category": "BU_Antrag", 
    "confidence": 0.89,
    "all_scores": {...}
  },
  "storage": {
    "sqlite": {"document_id": "doc_123"},
    "pinecone": {"chunks_stored": 5}
  },
  "processing_job_id": "uuid-here"
}
```

## Konfiguration

### Environment Variables
```bash
# Storage Backends
BU_VECTOR_DB__ENABLE_VECTOR_DB=true
EMBED_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
EMBED_DIM=768

# Pinecone Configuration
PINECONE_API_KEY=your-key
PINECONE_INDEX=bu-processor-embeddings
PINECONE_ENVIRONMENT=us-west1-gcp

# API Authentication
BU_API_TOKEN=dev-local-api-key  # Client-side
BU_API_KEY=dev-local-api-key    # Server-side
```

## Monitoring & Observability

### Structured Logging
```python
logger.info("Job completed successfully", 
           job_id=job_id,
           filename=filename,
           classification=result['predicted_label'],
           processing_time=duration)
```

### Health Checks
- Job queue status
- Storage backend connectivity
- Model availability
- Processing statistics

## Production Considerations

### Current (MVP): FastAPI BackgroundTasks
- ✅ Simple, keine externen Dependencies
- ✅ Ausreichend für moderate Lasten
- ⚠️ Jobs gehen bei Server-Restart verloren
- ⚠️ Keine Horizontal-Skalierung

### Zukunft: Celery + Redis
```python
# Upgrade-Pfad für Production
@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3})
def process_pdf_task(self, file_path: str, filename: str):
    return process_pdf(file_path, store_in_pinecone=True, store_in_sqlite=True)
```

### Skalierung
- **Queue Management**: Redis/RabbitMQ für Job Persistence
- **Worker Pools**: Separate Worker-Instanzen
- **Load Balancing**: Horizontale API-Skalierung
- **Monitoring**: Prometheus + Grafana

## Testing

### API Tests
```bash
# Upload und Status-Tracking
curl -X POST http://localhost:8000/ingest/pdf \
  -H "Authorization: Bearer dev-local-api-key" \
  -F "file=@test.pdf"

curl http://localhost:8000/ingest/status/job-id \
  -H "Authorization: Bearer dev-local-api-key"
```

### CLI Tests
```bash
# Verschiedene Modi testen
python scripts/process_pdf_cli.py test_bu_document.pdf --verbose
python scripts/process_pdf_cli.py --batch data/pdfs/ --output results/
```

## Integration Benefits

1. **Einheitliche Logik**: API und CLI nutzen dieselbe `process_pdf()` Funktion
2. **Konsistente Results**: Identische Verarbeitung und Metadaten
3. **Robuste Error Handling**: Retry-Logik und strukturierte Fehlerbehandlung  
4. **Skalierbare Architektur**: MVP → Production ready
5. **Monitoring Ready**: Strukturierte Logs und Job-Status

## Next Steps

1. **Load Testing**: Simuliere hohe Upload-Frequenz
2. **Queue Persistence**: Redis Integration für Job-Persistence
3. **Worker Scaling**: Separate Worker-Prozesse
4. **Monitoring Dashboard**: Job-Statistics und Health-Checks
5. **Webhook Notifications**: Completion Callbacks
