@echo off
REM =====================================================
REM PINECONE SMOKE TEST LAUNCHER
REM =====================================================
REM Quick launcher for the Pinecone smoke test
REM Sets up environment and runs the test

echo 🧪 PINECONE SMOKE TEST LAUNCHER
echo =====================================

REM Check if we're in the right directory
if not exist "bu_processor" (
    echo ❌ Error: bu_processor directory not found
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python not found in PATH
    echo Please install Python or add it to your PATH
    pause
    exit /b 1
)

REM Check for .env file
if exist ".env" (
    echo ✅ Loading environment from .env file
    REM Note: Windows doesn't natively support .env files
    REM You may need to set environment variables manually
) else (
    echo ⚠️  Warning: No .env file found
    echo Please copy .env.smoke.example to .env and configure:
    echo   - PINECONE_API_KEY
    echo   - PINECONE_ENV (v2) or PINECONE_CLOUD/PINECONE_REGION (v3)
    echo   - PINECONE_INDEX_NAME (optional)
    echo.
)

REM Navigate to project root and run the test
echo 🚀 Running Pinecone smoke test...
echo.

python scripts\pinecone_smoke.py

if errorlevel 1 (
    echo.
    echo ❌ Smoke test failed
    pause
    exit /b 1
) else (
    echo.
    echo ✅ Smoke test completed successfully
    pause
    exit /b 0
)
