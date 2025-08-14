@echo off
echo Fixing GitHub connection and switching to main branch...

REM Check current remote
echo Current remotes:
git remote -v

REM Remove existing origin if exists
git remote remove origin 2>nul

REM Add your GitHub repo (replace with your actual repo URL)
echo.
echo Please enter your GitHub repository URL (e.g., https://github.com/username/repo.git):
set /p REPO_URL="Repository URL: "
git remote add origin %REPO_URL%

REM Fetch all branches from remote
echo.
echo Fetching from remote...
git fetch origin

REM Check if main branch exists on remote
git ls-remote --heads origin main >nul 2>&1
if %errorlevel% equ 0 (
    echo Main branch found on remote
    
    REM Switch to main branch
    git checkout -b main origin/main 2>nul || git checkout main
    
    REM Set upstream
    git branch --set-upstream-to=origin/main main
    
    echo Successfully switched to main branch
) else (
    echo Main branch not found, checking for master...
    git ls-remote --heads origin master >nul 2>&1
    if %errorlevel% equ 0 (
        echo Master branch found, creating main from master...
        git checkout -b main origin/master
        git push origin main
        git branch --set-upstream-to=origin/main main
    ) else (
        echo No main or master branch found, creating new main...
        git checkout -b main
        git push -u origin main
    )
)

REM Show current status
echo.
echo Current branch and status:
git branch -a
git status

echo.
echo Git reconnection complete! You can now push with: git push
pause
