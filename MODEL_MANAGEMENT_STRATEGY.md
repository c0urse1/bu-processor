# 🤖 MODEL MANAGEMENT STRATEGY - VERSIONED ARTIFACTS
===============================================

## Übersicht

Versionierte Modell-Artefakte für stabiles, reproduzierbares und offline-fähiges ML-Deployment.

## 🎯 Ziele

- ✅ **Startup-Sicherheit**: Garantiert verfügbare Modelle beim Service-Start
- ✅ **Reproduzierbarkeit**: Identische Modelle über Environments hinweg
- ✅ **Keine Silent Updates**: Explizite Kontrolle über Modell-Versionen
- ✅ **Offline-Fähigkeit**: Funktioniert ohne externe API-Abhängigkeiten
- ✅ **Blue/Green Deployment**: Sichere Modell-Updates

## � Artifact-Struktur

```
artifacts/
├── model-v1/              # Production Model v1
│   ├── config.json        # Model configuration
│   ├── tokenizer.json     # Tokenizer
│   ├── tokenizer_config.json
│   ├── vocab.txt          # Vocabulary
│   ├── pytorch_model.bin  # Model weights
│   ├── labels.txt         # BU-specific labels
│   └── artifact_metadata.json
├── model-v2/              # Staging/Next version
└── model-experimental/    # R&D versions
```

## 🔧 Konfiguration

### Environment Variables (.env)

```bash
# PRODUCTION: Use stable versioned artifacts
ML_MODEL_REF=local:artifacts/model-v1

# DEVELOPMENT: Use HuggingFace directly (requires internet)
# ML_MODEL_REF=hf:deepset/gbert-base

# STAGING: Test new version
# ML_MODEL_REF=local:artifacts/model-v2
```

### Schema Support

| Schema | Beispiel | Beschreibung |
|--------|----------|--------------|
| `local:` | `local:artifacts/model-v1` | Lokales versioniertes Artifact |
| `hf:` | `hf:deepset/gbert-base` | HuggingFace Hub Model |
| `hf:` | `hf:my-org/private@v1.0.0` | HF mit spezifischer Version/Tag |

## �🚀 Deployment Workflow

### 1. Artifact Erstellen

```bash
# Neues Artifact von HuggingFace Model erstellen
python scripts/create_model_artifact.py --model deepset/gbert-base --version v1

# Oder mit custom model
python scripts/create_model_artifact.py --model my-org/bu-classifier --version v2
```

### 2. Testing auf Staging

```bash
# .env.staging
ML_MODEL_REF=local:artifacts/model-v2

# Test deployment
python -c "from bu_processor.pipeline.classifier_loader import load_classifier; load_classifier('local:artifacts/model-v2')"
```

### 3. Blue/Green Production Rollout

```bash
# Step 1: Deploy new artifact to production servers
cp -r artifacts/model-v2 /production/artifacts/

# Step 2: Update production configuration
# .env.production
ML_MODEL_REF=local:artifacts/model-v2

# Step 3: Restart services (zero-downtime with load balancer)
systemctl restart bu-processor

# Step 4: Verify & monitor
curl http://api/models/info
```

### 4. Rollback (falls nötig)

```bash
# Immediate rollback to previous version
# .env.production  
ML_MODEL_REF=local:artifacts/model-v1

systemctl restart bu-processor
```

## 📊 Robust Model Loader

### Schema-Based Loading
```python
from bu_processor.pipeline.classifier_loader import load_classifier

# Robust loading with clear error handling
tokenizer, model = load_classifier("local:artifacts/model-v1")
```

### Model Information
```python
from bu_processor.pipeline.classifier_loader import get_model_info

info = get_model_info("local:artifacts/model-v1")
print(info)
# {
#   "model_ref": "local:artifacts/model-v1",
#   "type": "local", 
#   "available": True,
#   "model_type": "bert",
#   "num_labels": 8
# }
```

### Integrated Labels
```python
from bu_processor.pipeline.classifier_loader import load_labels

labels = load_labels("local:artifacts/model-v1")
# ['BU_Bedingungswerk', 'BU_Antrag', 'BU_Risikopruefung', ...]
```

## 🔍 Quality & Validation

### Artifact Creation & Testing
```bash
# Create versioned artifact
python scripts/create_model_artifact.py --model deepset/gbert-base --version v1

# Validate artifact structure
python -c "
from bu_processor.pipeline.classifier_loader import get_model_info
info = get_model_info('local:artifacts/model-v1')
print('✅ Available:', info['available'])
print('📊 Model type:', info.get('model_type'))
"
```

### Integration Testing
```bash
# Test model loading in real classifier
python -c "
from bu_processor.ml.classifier import RealMLClassifier
clf = RealMLClassifier()
print('✅ Classifier loads with versioned model')
"
```

## 📈 Benefits Over Previous Approach

### Before (Fragile)
```python
# Fragile directory-based checks
if Path("artifacts/model-v1").exists():
    model = AutoModel.from_pretrained("artifacts/model-v1")
else:
    model = AutoModel.from_pretrained("deepset/gbert-base")  # Silent fallback
```

### After (Robust)
```python
# Explicit, validated loading
model_ref = os.getenv("ML_MODEL_REF", "local:artifacts/model-v1")
tokenizer, model = load_classifier(model_ref)  # Clear errors if fails
```

### Advantages

| Aspect | Before | After |
|--------|--------|-------|
| **Startup** | ❌ Fails if HF down | ✅ Reliable with local artifacts |
| **Versioning** | ❌ Implicit | ✅ Explicit version control |
| **Errors** | ❌ Silent fallbacks | ✅ Clear error messages |
| **Testing** | ❌ Hard to reproduce | ✅ Identical artifacts everywhere |
| **Deployment** | ❌ Risky updates | ✅ Blue/Green with rollback |

## 🛠️ Integration Points

### RealMLClassifier Update
```python
# Replace existing model loading logic
class RealMLClassifier:
    def __init__(self):
        model_ref = os.getenv("ML_MODEL_REF", "local:artifacts/model-v1")
        self.tokenizer, self.model = load_classifier(model_ref)
        self.labels = load_labels(model_ref)
```

### Configuration Schema
```python
# Add to core/config.py
class MLConfig(BaseModel):
    model_ref: str = Field(
        default="local:artifacts/model-v1",
        description="Model reference (local:path or hf:name[@version])"
    )
```

## 🔄 Versioning Strategy

- **model-v1**: Baseline production model
- **model-v2**: Major updates, new training data
- **model-v1.1**: Hotfixes, label corrections
- **model-experimental**: R&D, feature testing

## 📋 Production Deployment Checklist

- [ ] **Create Artifact**: `python scripts/create_model_artifact.py`
- [ ] **Local Validation**: Test loading & basic inference
- [ ] **Staging Deploy**: Update `ML_MODEL_REF` in staging env
- [ ] **Integration Tests**: Full API test suite
- [ ] **Performance Tests**: Benchmark vs. current production
- [ ] **Production Deploy**: Blue/Green rollout
- [ ] **Monitor Metrics**: Error rates, latency, accuracy
- [ ] **Document Change**: Update version changelog

## 🎯 Ready for Production

The versioned artifact strategy provides:
- ✅ **Rock-solid reliability** with local model artifacts
- ✅ **Explicit version control** for all environments  
- ✅ **Safe deployment processes** with Blue/Green rollouts
- ✅ **Clear error handling** instead of silent fallbacks
- ✅ **Offline operation** for maximum uptime

Ready to implement in production! 🚀

## Problem: Large Model Files (255+ MB)
GitHub blocks files >100MB, making model deployment challenging.

## ✅ **Solution: Dynamic Model Loading**

### 1. **Development Setup:**
```bash
# Use HuggingFace models directly (no local storage)
export ML_MODEL_REF="hf:deepset/gbert-base"
```

### 2. **Production Deployment:**
```bash
# Download on first startup
python scripts/download_model.py setup

# Or specific model
python scripts/download_model.py download deepset/gbert-base
```

### 3. **Docker Deployment:**
```dockerfile
# In Dockerfile
COPY scripts/download_model.py /app/scripts/
RUN python scripts/download_model.py setup
```

### 4. **Kubernetes/Cloud:**
```yaml
# Use init containers or persistent volumes
initContainers:
- name: model-downloader
  image: bu-processor:latest
  command: ["python", "scripts/download_model.py", "setup"]
```

## 🔧 **Configuration Options:**

### **Local Development (No Large Files):**
```env
ML_MODEL_REF=hf:deepset/gbert-base
```

### **Production (Pre-downloaded):**
```env
ML_MODEL_REF=local:artifacts/model-v1
```

### **Cloud Storage (Advanced):**
```env
ML_MODEL_REF=s3://your-bucket/models/gbert-base
ML_MODEL_REF=gs://your-bucket/models/gbert-base
```

## 📊 **Benefits:**
- ✅ No large files in Git repository
- ✅ Flexible model switching
- ✅ Production-ready deployment
- ✅ Automatic model downloading
- ✅ Version-controlled configuration

## 🚨 **Quick Fix for Current Issue:**
```bash
# Remove large file from Git history
git rm --cached artifacts/model-v1/model.safetensors
git commit -m "Remove large model file - use dynamic loading"

# Use HuggingFace model instead
echo "ML_MODEL_REF=hf:deepset/gbert-base" >> .env
```
