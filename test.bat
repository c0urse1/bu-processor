@echo off
REM Windows Batch Script für BU-Processor Tests

setlocal enabledelayedexpansion

echo ========================================
echo BU-PROCESSOR TEST RUNNER (Windows)
echo ========================================

REM Check if Python available
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ❌ Python nicht gefunden. Installiere Python 3.8+ 
    exit /b 1
)

REM Check if in correct directory
if not exist "pyproject.toml" (
    echo ❌ Nicht im Projekt-Root. Navigiere zu bu_processor/ Verzeichnis.
    exit /b 1
)

REM Parse command line arguments
set COMMAND=%1
if "%COMMAND%"=="" set COMMAND=all

echo 🎯 Test-Modus: %COMMAND%

REM Install dependencies if needed
if "%COMMAND%"=="install" (
    echo 📦 Installiere Test-Dependencies...
    pip install -r requirements-dev.txt
    if !errorlevel! equ 0 (
        echo ✅ Dependencies installiert
    ) else (
        echo ❌ Dependencies-Installation fehlgeschlagen
        exit /b 1
    )
    exit /b 0
)

REM Validate setup
if "%COMMAND%"=="validate" (
    echo 🔍 Validiere Test-Setup...
    python scripts\run_tests.py validate
    exit /b !errorlevel!
)

REM Run specific test categories
if "%COMMAND%"=="all" (
    echo 🧪 Führe alle Tests aus...
    python -m pytest -v --cov=bu_processor --cov-report=term-missing tests\
) else if "%COMMAND%"=="unit" (
    echo 🔧 Führe Unit Tests aus...
    python -m pytest -v -m unit tests\
) else if "%COMMAND%"=="integration" (
    echo 🔗 Führe Integration Tests aus...
    python -m pytest -v -m integration tests\
) else if "%COMMAND%"=="quick" (
    echo ⚡ Führe schnelle Tests aus...
    python -m pytest -v -m "not slow" tests\
) else if "%COMMAND%"=="mock" (
    echo 🎭 Führe Mock-Tests aus...
    python -m pytest -v -m mock tests\
) else if "%COMMAND%"=="coverage" (
    echo 📊 Führe Tests mit Coverage Report aus...
    python -m pytest -v --cov=bu_processor --cov-report=html:htmlcov --cov-report=term-missing tests\
    echo 📈 Coverage Report: htmlcov\index.html
) else if "%COMMAND%"=="classifier" (
    echo 🤖 Teste Classifier...
    python -m pytest -v tests\test_classifier.py
) else if "%COMMAND%"=="pdf" (
    echo 📄 Teste PDF-Extraktor...
    python -m pytest -v tests\test_pdf_extractor.py
) else if "%COMMAND%"=="pipeline" (
    echo 🔄 Teste Pipeline-Komponenten...
    python -m pytest -v tests\test_pipeline_components.py
) else if "%COMMAND%"=="performance" (
    echo 🏃 Führe Performance Tests aus...
    python -m pytest -v --durations=20 -k "performance or batch or timing" tests\
) else if "%COMMAND%"=="ci" (
    echo 🚀 CI/CD Test-Modus...
    python -m pytest -v --cov=bu_processor --cov-report=xml:coverage.xml --cov-fail-under=70 tests\
) else (
    echo ❓ Unbekanntes Kommando: %COMMAND%
    echo.
    echo Verfügbare Kommandos:
    echo   install     - Installiere Test-Dependencies
    echo   validate    - Validiere Test-Setup
    echo   all         - Alle Tests
    echo   unit        - Nur Unit Tests
    echo   integration - Nur Integration Tests
    echo   quick       - Nur schnelle Tests
    echo   mock        - Nur Mock-Tests
    echo   coverage    - Tests mit Coverage Report
    echo   classifier  - Nur Classifier Tests
    echo   pdf         - Nur PDF-Extraktor Tests
    echo   pipeline    - Nur Pipeline Tests
    echo   performance - Performance Tests
    echo   ci          - CI/CD Modus
    echo.
    echo Beispiele:
    echo   test.bat all
    echo   test.bat quick
    echo   test.bat classifier
    echo   test.bat coverage
    exit /b 1
)

REM Check test result
if !errorlevel! equ 0 (
    echo.
    echo ✅ Tests erfolgreich abgeschlossen!
) else (
    echo.
    echo ❌ Tests fehlgeschlagen (Code: !errorlevel!^)
    echo 💡 Prüfe Logs für Details
)

exit /b !errorlevel!
