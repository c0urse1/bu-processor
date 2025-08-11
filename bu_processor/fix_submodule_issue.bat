@echo off
echo Fixing Git Submodule Issue...

echo.
echo 1. Current directory structure:
dir /b

echo.
echo 2. Checking git status:
git status

echo.
echo 3. Checking for .gitmodules file:
if exist .gitmodules (
    echo .gitmodules found:
    type .gitmodules
) else (
    echo No .gitmodules file found
)

echo.
echo 4. Removing submodule entry if exists...
git rm --cached bu_processor 2>nul
git config --remove-section submodule.bu_processor 2>nul

echo.
echo 5. Removing .gitmodules if it references bu_processor...
if exist .gitmodules (
    findstr /v "bu_processor" .gitmodules > .gitmodules.tmp 2>nul
    move .gitmodules.tmp .gitmodules 2>nul
    if exist .gitmodules (
        for %%A in (.gitmodules) do if %%~zA==0 del .gitmodules
    )
)

echo.
echo 6. Re-adding bu_processor as normal directory...
git add bu_processor/

echo.
echo 7. Current status:
git status

echo.
echo 8. Committing fix...
git commit -m "Fix: Remove bu_processor submodule, add as normal directory"

echo.
echo 9. Pushing to GitHub...
git push

echo.
echo Submodule issue should be fixed!
pause
