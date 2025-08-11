@echo off
echo Fixing Git Submodule Issue from main directory...

cd /d C:\ml_classifier_poc

echo.
echo Current directory: %CD%

echo.
echo 1. Checking git status:
git status

echo.
echo 2. Checking for .gitmodules file:
if exist .gitmodules (
    echo .gitmodules found:
    type .gitmodules
    echo.
    echo Removing bu_processor from .gitmodules...
    findstr /v "bu_processor" .gitmodules > .gitmodules.tmp 2>nul
    move .gitmodules.tmp .gitmodules 2>nul
    for %%A in (.gitmodules) do if %%~zA==0 del .gitmodules
) else (
    echo No .gitmodules file found
)

echo.
echo 3. Removing submodule cache entry...
git rm --cached bu_processor 2>nul

echo.
echo 4. Removing submodule config...
git config --remove-section submodule.bu_processor 2>nul

echo.
echo 5. Re-adding bu_processor as normal directory...
git add bu_processor/

echo.
echo 6. Current status:
git status

echo.
echo 7. Committing fix...
git commit -m "Fix: Remove bu_processor submodule, add as normal directory"

echo.
echo 8. Pushing to GitHub...
git push

echo.
echo Fixed! bu_processor should now be a normal directory.
pause
