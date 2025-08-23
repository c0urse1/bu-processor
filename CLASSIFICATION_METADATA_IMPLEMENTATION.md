# 🏷️ KLASSIFIKATION IM ARTEFAKT UND METADATEN

## Übersicht

Das BU-Processor System speichert jetzt Klassifikationsergebnisse deterministisch in Artefakten und strukturiert in Metadaten für erweiterte Filterung und Analyse.

## Deterministische Labels aus Artefakt

### Labels.txt im Model-Artefakt
```bash
# artifacts/model-v1/labels.txt
BU_Bedingungswerk
BU_Antrag
BU_Risikopruefung
BU_Leitfaden
BU_Fallbeispiel
BU_FAQ
BU_Presse
Sonstiges
```

### Automatisches Label-Loading
```python
# In RealMLClassifier
labels_path = os.path.join(model_path, "labels.txt")
if os.path.exists(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        self.labels = [line.strip() for line in f if line.strip()]
```

### Deterministische Klassifikation
- ✅ Labels sind fest im Artefakt verankert
- ✅ Keine Abhängigkeit von externen Konfigurationen
- ✅ Konsistente category_label Ausgabe
- ✅ Versionierte Label-Sets pro Modell-Artefakt

## Metadaten-Struktur für Filterung

### Core Metadata Fields
Direkt in Root-Level für einfache Filterung:
```json
{
  "predicted_label": "BU_Antrag",
  "predicted_category": "BU_Antrag", 
  "predicted_confidence": 0.89,
  // ... andere Felder
}
```

### Detaillierte Classification Metadata
Vollständige Informationen im `classification` Objekt:
```json
{
  "classification": {
    "predicted_label": "BU_Antrag",
    "predicted_category": "BU_Antrag",
    "confidence": 0.89,
    "all_scores": {
      "BU_Antrag": 0.89,
      "BU_Bedingungswerk": 0.08,
      "Sonstiges": 0.03
    },
    "text_length": 1500,
    "model_labels": [
      "BU_Bedingungswerk", "BU_Antrag", "BU_Risikopruefung",
      "BU_Leitfaden", "BU_Fallbeispiel", "BU_FAQ", 
      "BU_Presse", "Sonstiges"
    ],
    "model_info": {
      "model_dir": "artifacts/model-v1",
      "labels_source": "artifact_labels.txt"
    }
  }
}
```

## Storage Implementation

### SQLite Storage
```python
# Enhanced metadata storage in SQLite
metadata = {
    # Core fields for filtering
    "predicted_label": classification_result["predicted_label"],
    "predicted_confidence": classification_result["confidence"],
    
    # Detailed classification info
    "classification": {
        "model_labels": classifier.get_available_labels(),
        "labels_source": "artifact_labels.txt"
    }
}

storage.add_document(content=text, metadata=metadata, source=filename)
```

### Pinecone Storage
```python
# Same metadata structure in Pinecone chunks
chunk_metadata = {
    **metadata,  # Includes predicted_label, predicted_confidence
    "chunk_index": i,
    "total_chunks": len(chunks),
    "chunk_text": chunk[:200] + "..."
}

pinecone_manager.upsert_document(
    doc_id=chunk_id,
    content=chunk,
    metadata=chunk_metadata
)
```

## Filtering & Analysis Capabilities

### SQLite Filtering
```python
# Filter by classification label
documents = storage.search_documents(
    predicted_label="BU_Antrag",
    limit=10
)

# Filter by confidence threshold
high_confidence_docs = storage.search_documents(
    min_confidence=0.8,
    limit=20
)

# Combined filtering
precise_antrag_docs = storage.search_documents(
    predicted_label="BU_Antrag",
    min_confidence=0.9,
    query="Versicherung"
)
```

### Classification Statistics
```python
# Get label distribution and confidence stats
stats = storage.get_classification_stats()
# Returns:
{
  "total_documents": 150,
  "labels": {
    "BU_Antrag": 45,
    "BU_Bedingungswerk": 30,
    "BU_FAQ": 25,
    "Sonstiges": 50
  },
  "confidence": {
    "mean": 0.87,
    "min": 0.45,
    "max": 0.99,
    "count": 150
  }
}
```

### Pinecone Metadata Filtering
```python
# Pinecone metadata filtering (wenn implementiert)
results = pinecone_index.query(
    vector=embedding,
    filter={
        "predicted_label": {"$eq": "BU_Antrag"},
        "predicted_confidence": {"$gte": 0.8}
    },
    top_k=10,
    include_metadata=True
)
```

## Code Integration Points

### Unified process_pdf Function
```python
# In bu_processor.ingest.process_pdf()
metadata = {
    # Core fields for filtering
    "predicted_label": classification_result["predicted_label"],
    "predicted_confidence": classification_result["confidence"],
    
    # Enhanced classification details
    "classification": {
        "model_labels": classifier.get_available_labels() or [],
        "model_info": {
            "labels_source": "artifact_labels.txt" if classifier.get_available_labels() else "fallback"
        }
    }
}
```

### API Endpoints
```python
# Both sync and async endpoints use consistent metadata
# /process/pdf (synchronous)
# /ingest/pdf (background job)

# Both store same metadata structure:
meta["predicted_label"] = prediction.label
meta["predicted_confidence"] = prediction.confidence
```

### Background Jobs
```python
# Background job system preserves metadata structure
job_metadata = {
    "predicted_label": classification_result["predicted_label"],
    "predicted_confidence": classification_result["confidence"],
    "processing_type": "background_job",
    "job_id": job.job_id
}
```

## Validation & Testing

### Label Validation Script
```bash
# Validate labels are loaded from artifacts
python scripts/validate_labels_metadata.py

# Check stored document metadata
python scripts/validate_labels_metadata.py --check-stored-docs
```

### Metadata Analysis Script
```bash
# Analyze classification metadata in storage
python scripts/check_classification_metadata.py

# Filter by specific label
python scripts/check_classification_metadata.py --filter-label BU_Antrag

# Filter by confidence threshold
python scripts/check_classification_metadata.py --min-confidence 0.8
```

## Schema Consistency

### Database Schema
```sql
-- SQLite documents table with JSON metadata
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT,
    source TEXT,
    created_at DATETIME,
    meta JSON  -- Contains predicted_label, predicted_confidence, classification
);

-- Query examples
SELECT * FROM documents 
WHERE JSON_EXTRACT(meta, '$.predicted_label') = 'BU_Antrag';

SELECT * FROM documents 
WHERE CAST(JSON_EXTRACT(meta, '$.predicted_confidence') AS REAL) >= 0.8;
```

### Pinecone Metadata Schema
```python
# Consistent metadata structure in Pinecone
{
    "predicted_label": "BU_Antrag",
    "predicted_confidence": 0.89,
    "filename": "antrag_2024.pdf",
    "chunk_index": 0,
    "total_chunks": 3,
    "classification": {...}  # Full classification details
}
```

## Migration Strategy

### Existing Documents
- Alte Dokumente ohne neue Metadaten-Felder bleiben kompatibel
- Migration-Script kann fehlende Felder nachträglich hinzufügen
- Filtering funktioniert mit NULL-Checks

### Model Artifact Updates
```bash
# Update model with new labels
python scripts/create_model_artifact.py --model deepset/gbert-base --version v2

# Update .env to use new artifact
ML_MODEL_REF=local:artifacts/model-v2
```

## Benefits

1. **Deterministische Labels**: Labels fest im Artefakt verankert
2. **Erweiterte Filterung**: Nach Label und Confidence filtern
3. **Analytik-Ready**: Statistiken und Trend-Analysen möglich
4. **API-kompatibel**: Konsistente Metadaten zwischen Sync/Async
5. **Storage-agnostisch**: Gleiche Struktur in SQLite und Pinecone
6. **Versionierte Labels**: Pro Artefakt-Version eigene Label-Sets

## Usage Examples

### Find All High-Confidence Anträge
```python
antraege = storage.search_documents(
    predicted_label="BU_Antrag",
    min_confidence=0.9
)
```

### Analyze Classification Distribution
```python
stats = storage.get_classification_stats()
for label, count in stats['labels'].items():
    print(f"{label}: {count} documents")
```

### Process with Metadata Tracking
```python
result = process_pdf(
    file_path="new_document.pdf",
    store_in_pinecone=True,
    store_in_sqlite=True
)

# Metadata automatically includes:
# - predicted_label from artifact labels
# - predicted_confidence
# - model_labels from labels.txt
# - labels_source = "artifact_labels.txt"
```
