@echo off
REM ============================================================================
REM PRE-COMMIT SETUP SCRIPT FOR BU-PROCESSOR (Windows)
REM ============================================================================
REM Automatische Installation und Konfiguration der Pre-Commit Hooks

echo 🚀 Setting up Pre-Commit Hooks for BU-Processor...
echo =================================================

REM Check if we're in the right directory
if not exist ".pre-commit-config.yaml" (
    echo ❌ Error: .pre-commit-config.yaml not found. Make sure you're in the project root.
    exit /b 1
)

REM Install development dependencies (includes pre-commit)
echo 📦 Installing development dependencies...
pip install -r requirements-dev.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    exit /b 1
)

REM Install pre-commit hooks
echo 🔧 Installing pre-commit hooks...
pre-commit install
if %errorlevel% neq 0 (
    echo ❌ Failed to install pre-commit hooks
    exit /b 1
)

REM Run pre-commit on all files to check initial setup
echo ✅ Running initial pre-commit check on all files...
pre-commit run --all-files
if %errorlevel% neq 0 (
    echo ⚠️  Some files needed formatting. This is normal for first setup.
    echo 💡 Files have been automatically formatted. Review and commit changes.
)

echo.
echo 🎉 Pre-commit setup complete!
echo =================================================
echo ✓ Pre-commit hooks installed
echo ✓ Code quality tools configured:
echo   - Black (code formatting)
echo   - isort (import organization)
echo   - Flake8 (linting)
echo   - MyPy (type checking)  
echo   - General file cleanup hooks
echo.
echo 📝 Next steps:
echo   - Hooks will run automatically before each commit
echo   - To manually run all hooks: pre-commit run --all-files
echo   - To skip hooks for a commit: git commit --no-verify
echo   - To update hook versions: pre-commit autoupdate
echo.
pause
