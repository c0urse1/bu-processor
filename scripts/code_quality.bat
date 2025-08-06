@echo off
REM Code Quality Script für Windows
REM ===============================
REM
REM Führt alle Code-Qualitäts-Tools in der richtigen Reihenfolge aus.
REM
REM Usage:
REM   scripts\code_quality.bat [check|fix|mypy-only]

echo.
echo ========================================
echo   BU-Processor Code Quality Check
echo ========================================
echo.

REM Determine mode
set MODE=%1
if "%MODE%"=="" set MODE=fix

echo Mode: %MODE%
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo ⚠️  Warnung: Kein Virtual Environment erkannt
    echo    Aktiviere mit: venv\Scripts\activate
    echo.
)

REM Install dev dependencies if needed
echo 🔧 Checking dev dependencies...
pip show black >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing dev dependencies...
    pip install -r requirements-dev.txt
    if errorlevel 1 (
        echo ❌ Failed to install dev dependencies
        exit /b 1
    )
    echo ✅ Dev dependencies installed
    echo.
)

REM Run quality checks based on mode
if "%MODE%"=="check" (
    echo 🔍 Running in CHECK mode - no changes will be made
    echo.
    
    REM 1. Check import sorting
    echo 🔧 Checking import sorting with isort...
    python -m isort --check --diff bu_processor tests scripts
    if errorlevel 1 (
        echo ❌ isort check failed
        echo 💡 Run: python -m isort bu_processor tests scripts
        set /a FAILED+=1
    ) else (
        echo ✅ isort check passed
    )
    echo.
    
    REM 2. Check code formatting
    echo 🔧 Checking code formatting with black...
    python -m black --check --diff bu_processor tests scripts
    if errorlevel 1 (
        echo ❌ black check failed
        echo 💡 Run: python -m black bu_processor tests scripts  
        set /a FAILED+=1
    ) else (
        echo ✅ black check passed
    )
    echo.
    
) else if "%MODE%"=="mypy-only" (
    echo 🔍 Running MYPY-ONLY mode
    echo.
    goto :mypy
    
) else (
    echo 🔧 Running in FIX mode - files will be modified
    echo.
    
    REM 1. Fix import sorting
    echo 🔧 Sorting imports with isort...
    python -m isort bu_processor tests scripts
    if errorlevel 1 (
        echo ❌ isort failed
        set /a FAILED+=1
    ) else (
        echo ✅ isort completed
    )
    echo.
    
    REM 2. Fix code formatting
    echo 🔧 Formatting code with black...
    python -m black bu_processor tests scripts
    if errorlevel 1 (
        echo ❌ black failed
        set /a FAILED+=1
    ) else (
        echo ✅ black completed
    )
    echo.
)

REM 3. Linting with flake8
echo 🔧 Linting with flake8...
python -m flake8 bu_processor tests scripts
if errorlevel 1 (
    echo ❌ flake8 found issues
    echo 💡 Fix manually based on flake8 output
    set /a FAILED+=1
) else (
    echo ✅ flake8 passed
)
echo.

:mypy
REM 4. Type checking with mypy
echo 🔧 Type checking with mypy...
python -m mypy bu_processor
if errorlevel 1 (
    echo ❌ mypy found type issues  
    echo 💡 Add type hints or adjust mypy configuration
    set /a FAILED+=1
) else (
    echo ✅ mypy passed
)
echo.

REM Summary
echo ========================================
if %FAILED% gtr 0 (
    echo ❌ %FAILED% tools reported issues
    echo.
    echo 💡 Common fixes:
    echo    - Run scripts\code_quality.bat fix
    echo    - Check flake8 output for style issues
    echo    - Add missing type hints for mypy
    echo    - Install missing dependencies
    echo.
    exit /b 1
) else (
    echo 🎉 All code quality checks passed!
    echo.
    echo ✨ Your code follows best practices:
    echo    - ✅ Imports properly sorted
    echo    - ✅ Code properly formatted  
    echo    - ✅ No linting issues
    echo    - ✅ Type hints validated
    echo.
)

exit /b 0
