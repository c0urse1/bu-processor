@echo off
echo Git Push from correct directory...

echo Changing to main git directory...
cd /d C:\ml_classifier_poc

echo.
echo Current directory: %CD%

echo.
echo 1. Checking current status...
git status

echo.
echo 2. Adding all changes...
git add .

echo.
echo 3. Committing changes...
set /p COMMIT_MSG="Enter commit message (or press Enter for default): "
if "%COMMIT_MSG%"=="" set COMMIT_MSG=Update code

git commit -m "%COMMIT_MSG%"

echo.
echo 4. Pushing to GitHub...
git push

echo.
echo 5. Final status:
git status

echo.
echo Done! Your code should now be on GitHub.
pause
