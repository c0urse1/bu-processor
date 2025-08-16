@echo off
REM CI Check Script for Test Hygiene (Windows)
REM Usage: scripts\check-test-hygiene.bat

echo 🧪 Checking test file placement hygiene...

REM Check for test files in package directory
set "FOUND_ISSUES=0"

for /r "bu_processor\bu_processor" %%f in (test_*.py *_test.py) do (
    if exist "%%f" (
        echo ❌ ERROR: Test file found in package directory: %%f
        set "FOUND_ISSUES=1"
    )
)

if "%FOUND_ISSUES%"=="1" (
    echo.
    echo Tests must be placed in:
    echo   - tests\ ^(global tests^)
    echo   - bu_processor\tests\ ^(package-specific tests outside main package^)
    echo.
    echo NOT in:
    echo   - bu_processor\bu_processor\ ^(main package directory^)
    exit /b 1
)

REM Check test directory structure
echo 📁 Verifying test directory structure...

if not exist "tests" (
    echo ⚠️ WARNING: Global tests\ directory not found
)

if not exist "bu_processor\tests" (
    echo ⚠️ WARNING: Package tests\ directory not found at bu_processor\tests\
)

echo ✅ Test placement hygiene check passed!

REM Count test files (basic counting)
echo 📊 Test directories verified
if exist "tests" echo   - Global tests directory: exists
if exist "bu_processor\tests" echo   - Package tests directory: exists
