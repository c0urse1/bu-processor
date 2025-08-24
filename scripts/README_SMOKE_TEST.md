# 🧪 Pinecone Smoke Test

Minimaler Test für die Pinecone Integration ohne komplexe Pipeline-Komponenten.

## Vorbereitung

1. **Environment Setup**:
   ```bash
   cp .env.smoke.example .env
   # Bearbeite .env und setze deine Pinecone Credentials
   ```

2. **Minimale .env Konfiguration**:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key_here
   
   # Für v2 (pod-based):
   PINECONE_ENV=us-west1-gcp
   
   # Für v3 (serverless) - stattdessen:
   # PINECONE_CLOUD=gcp
   # PINECONE_REGION=us-west1
   
   PINECONE_INDEX_NAME=bu-processor
   PINECONE_NAMESPACE=bu
   ```

## Ausführung

### Windows:
```cmd
scripts\run_pinecone_smoke.bat
```

### Linux/macOS:
```bash
python scripts/pinecone_smoke.py
```

## Was wird getestet

1. ✅ **Embedder Initialisierung** - Lädt SentenceTransformers Modell
2. ✅ **PineconeManager Setup** - Verbindung zu Pinecone (v2/v3 kompatibel)
3. ✅ **Index Management** - Erstellt Index falls nicht vorhanden
4. ✅ **Vector Upsert** - Speichert Test-Embeddings
5. ✅ **Query Test** - Sucht ähnliche Dokumente

## Erwartete Ausgabe

```
🧪 PINECONE SMOKE TEST
==================================================
🔧 Initializing Embedder...
✅ Embedder initialized (dimension: 768)

🔧 Initializing PineconeManager...
✅ PineconeManager initialized

🔧 Ensuring index exists (dimension: 768)...
✅ Index ready

📝 Preparing test data...
🧮 Generating embeddings...
✅ Generated 2 embeddings

⬆️ Upserting vectors to Pinecone...
✅ Vectors upserted successfully

🔍 Testing query...

🎯 QUERY RESULT:
------------------------------
Query: 'Wann zahlt BU?'
Results found: 2

Result 1:
  ID: test-1
  Score: 0.7234
  Metadata: {'doc_id': 'smoke-doc'}

Result 2:
  ID: test-2
  Score: 0.6891
  Metadata: {'doc_id': 'smoke-doc'}

🎉 SMOKE TEST COMPLETED SUCCESSFULLY!
```

## Troubleshooting

- **Import Errors**: Starte aus dem Projekt-Root (`c:\ml_classifier_poc`)
- **API Key Fehler**: Prüfe `PINECONE_API_KEY` in `.env`
- **Environment Fehler**: Prüfe `PINECONE_ENV` (v2) oder `PINECONE_CLOUD`/`PINECONE_REGION` (v3)
- **Dimension Conflicts**: Index wird automatisch mit korrekter Dimension erstellt
