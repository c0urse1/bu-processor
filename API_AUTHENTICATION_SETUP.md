# ============================================================================
# API AUTHENTICATION SETUP GUIDE
# ============================================================================

# 🔍 ERGEBNIS DER REPO-ANALYSE:
# 
# Der BU-Processor Server erwartet:
# - ENV-Variable: BU_API_KEY 
# - Header: Authorization: Bearer <your-token>
#
# Gefunden in:
# - bu_processor/core/config.py: env_prefix="BU_" + api_key Field
# - bu_processor/api/main.py: verify_api_key() Funktion

# ============================================================================
# SCHRITT 1: SERVER-SIDE API KEY SETZEN
# ============================================================================

# Option A: .env Datei erstellen/erweitern
echo BU_API_KEY=your-secret-token-here >> .env

# Option B: Environment direkt setzen (Windows cmd)
set BU_API_KEY=your-secret-token-here

# Option C: Environment direkt setzen (PowerShell)
$env:BU_API_KEY = "your-secret-token-here"

# ============================================================================
# SCHRITT 2: CLIENT-SIDE VALIDATION SCRIPT KONFIGURIEREN  
# ============================================================================

# Das validate_classification.py Script nutzt BU_API_TOKEN
# Also beide Variablen auf denselben Wert setzen:

# Windows cmd:
set BU_API_KEY=your-secret-token-here
set BU_API_TOKEN=your-secret-token-here

# PowerShell:
$env:BU_API_KEY = "your-secret-token-here"
$env:BU_API_TOKEN = "your-secret-token-here"

# .env Datei:
BU_API_KEY=your-secret-token-here
BU_API_TOKEN=your-secret-token-here

# ============================================================================
# SCHRITT 3: SERVER STARTEN & TESTEN
# ============================================================================

# Server mit .env starten:
cd bu_processor
python -m uvicorn bu_processor.api.main:app --reload --env-file ../.env

# Oder mit Environment-Variablen:
set BU_API_KEY=your-secret-token-here && python -m uvicorn bu_processor.api.main:app --reload

# ============================================================================
# SCHRITT 4: SWAGGER UI TESTEN
# ============================================================================

# 1. Öffne: http://127.0.0.1:8000/docs
# 2. Klicke "Authorize" Button (🔒)
# 3. Trage ein: your-secret-token-here
# 4. Teste einen Endpoint

# ============================================================================
# SCHRITT 5: VALIDATION SCRIPT TESTEN
# ============================================================================

# Mit korrekten Environment-Variablen:
set BU_API_TOKEN=your-secret-token-here
python scripts/validate_classification.py

# Expected Output:
# 🔑 Using API token: ✅ Set
# 🔍 Checking API availability at http://127.0.0.1:8000...
# ✅ API is available at http://127.0.0.1:8000
# [INFO] Klassifiziere: document.pdf
#    ✅ Business Unit Report (confidence: 0.89)

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# Fehler 401: "API key required"
# → BU_API_KEY auf Server-Seite nicht gesetzt

# Fehler 403: "Invalid API key"  
# → BU_API_KEY (Server) ≠ BU_API_TOKEN (Client)

# Fehler "🔑 Using API token: ❌ Not set"
# → BU_API_TOKEN auf Client-Seite nicht gesetzt
