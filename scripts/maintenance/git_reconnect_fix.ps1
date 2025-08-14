# Git Repository Reconnection PowerShell Script
Write-Host "Fixing GitHub connection and switching to main branch..." -ForegroundColor Green

# Check current remote
Write-Host "`nCurrent remotes:" -ForegroundColor Yellow
git remote -v

# Remove existing origin if exists
Write-Host "`nRemoving existing origin..." -ForegroundColor Yellow
git remote remove origin 2>$null

# Add your GitHub repo
$repoUrl = Read-Host "`nPlease enter your GitHub repository URL (e.g., https://github.com/username/repo.git)"
git remote add origin $repoUrl

# Fetch all branches from remote
Write-Host "`nFetching from remote..." -ForegroundColor Yellow
git fetch origin

# Check if main branch exists on remote
$mainExists = git ls-remote --heads origin main 2>$null
if ($mainExists) {
    Write-Host "Main branch found on remote" -ForegroundColor Green
    
    # Switch to main branch
    git checkout -b main origin/main 2>$null
    if ($LASTEXITCODE -ne 0) {
        git checkout main
    }
    
    # Set upstream
    git branch --set-upstream-to=origin/main main
    
    Write-Host "Successfully switched to main branch" -ForegroundColor Green
} else {
    Write-Host "Main branch not found, checking for master..." -ForegroundColor Yellow
    $masterExists = git ls-remote --heads origin master 2>$null
    if ($masterExists) {
        Write-Host "Master branch found, creating main from master..." -ForegroundColor Yellow
        git checkout -b main origin/master
        git push origin main
        git branch --set-upstream-to=origin/main main
    } else {
        Write-Host "No main or master branch found, creating new main..." -ForegroundColor Yellow
        git checkout -b main
        git push -u origin main
    }
}

# Show current status
Write-Host "`nCurrent branch and status:" -ForegroundColor Yellow
git branch -a
git status

Write-Host "`nGit reconnection complete! You can now push with: git push" -ForegroundColor Green
Read-Host "Press Enter to continue"
