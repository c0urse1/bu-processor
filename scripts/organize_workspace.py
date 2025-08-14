#!/usr/bin/env python3
"""
Workspace Organization Script for BU-Processor
=============================================
Organizes the entire workspace into a clean, standardized structure.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

def print_status(message: str, success: bool = True):
    """Print colored status message."""
    emoji = "✅" if success else "❌"
    print(f"{emoji} {message}")

def create_directories(project_root: Path) -> None:
    """Create standard project directories."""
    print("🏗️ Creating standard directory structure...")
    
    directories = [
        "docs/api",
        "docs/guides", 
        "docs/implementation",
        "examples",
        "scripts/maintenance",
        "temp",
        ".github/workflows"
    ]
    
    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print_status(f"Directory: {dir_path}")

def move_documentation(project_root: Path) -> None:
    """Move documentation files to proper structure."""
    print("\n📚 Organizing documentation...")
    
    # Documentation files in bu_processor that should be moved
    doc_moves = {
        # Implementation documentation
        "IMPLEMENTATION_COMPLETE_SUMMARY.md": "docs/implementation/",
        "FINAL_IMPLEMENTATION_SUMMARY.md": "docs/implementation/",
        "SESSION_COMPLETE_SUMMARY.md": "docs/implementation/",
        "CONFIDENCE_COMPLETION_SUMMARY.md": "docs/implementation/",
        "HEALTH_CHECK_COMPLETION_SUMMARY.md": "docs/implementation/",
        
        # Development guides
        "CONTRIBUTING.md": "docs/guides/",
        "CODE_QUALITY.md": "docs/guides/",
        "GIT_REPAIR_GUIDE.md": "docs/guides/",
        "SUBMODULE_FIX_GUIDE.md": "docs/guides/",
        
        # Completion markers (move to implementation)
        "*_COMPLETE.md": "docs/implementation/",
        "*_COMPLETE*.md": "docs/implementation/",
    }
    
    bu_processor_dir = project_root / "bu_processor"
    
    for pattern, target_dir in doc_moves.items():
        if "*" in pattern:
            # Handle glob patterns
            for file_path in bu_processor_dir.glob(pattern):
                if file_path.is_file():
                    target_path = project_root / target_dir / file_path.name
                    try:
                        shutil.move(str(file_path), str(target_path))
                        print_status(f"Moved: {file_path.name} → {target_dir}")
                    except Exception as e:
                        print_status(f"Failed to move {file_path.name}: {e}", success=False)
        else:
            # Handle specific files
            source_path = bu_processor_dir / pattern
            if source_path.exists():
                target_path = project_root / target_dir / source_path.name
                try:
                    shutil.move(str(source_path), str(target_path))
                    print_status(f"Moved: {pattern} → {target_dir}")
                except Exception as e:
                    print_status(f"Failed to move {pattern}: {e}", success=False)

def clean_cache_files(project_root: Path) -> None:
    """Clean all cache and temporary files."""
    print("\n🧹 Cleaning cache and temporary files...")
    
    # Remove Python cache directories
    for cache_dir in project_root.rglob("__pycache__"):
        if cache_dir.is_dir():
            try:
                shutil.rmtree(cache_dir)
                print_status(f"Removed: {cache_dir.relative_to(project_root)}")
            except Exception as e:
                print_status(f"Failed to remove {cache_dir}: {e}", success=False)
    
    # Remove Python cache files
    for cache_file in project_root.rglob("*.pyc"):
        try:
            cache_file.unlink()
            print_status(f"Removed: {cache_file.relative_to(project_root)}")
        except Exception as e:
            print_status(f"Failed to remove {cache_file}: {e}", success=False)
    
    # Remove pytest cache
    pytest_cache = project_root / ".pytest_cache"
    if pytest_cache.exists():
        try:
            shutil.rmtree(pytest_cache)
            print_status("Removed: .pytest_cache")
        except Exception as e:
            print_status(f"Failed to remove .pytest_cache: {e}", success=False)
    
    # Remove coverage files
    coverage_files = [".coverage", "htmlcov"]
    for item in coverage_files:
        path = project_root / item
        if path.exists():
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                print_status(f"Removed: {item}")
            except Exception as e:
                print_status(f"Failed to remove {item}: {e}", success=False)

def organize_scripts(project_root: Path) -> None:
    """Organize scripts into proper categories."""
    print("\n⚙️ Organizing scripts...")
    
    bu_processor_dir = project_root / "bu_processor"
    scripts_dir = project_root / "scripts"
    maintenance_dir = scripts_dir / "maintenance"
    
    # Scripts that should be moved to maintenance
    maintenance_scripts = [
        "check_deps.py",
        "final_migration_summary.py",
        "solution_summary.py",
        "fix_submodule_issue.bat",
        "git_push_easy.bat",
        "git_reconnect_fix.bat",
        "git_reconnect_fix.ps1",
    ]
    
    for script_name in maintenance_scripts:
        source_path = bu_processor_dir / script_name
        if source_path.exists():
            target_path = maintenance_dir / script_name
            try:
                shutil.move(str(source_path), str(target_path))
                print_status(f"Moved script: {script_name} → scripts/maintenance/")
            except Exception as e:
                print_status(f"Failed to move {script_name}: {e}", success=False)

def create_environment_files(project_root: Path) -> None:
    """Create standard environment and configuration files."""
    print("\n🔧 Creating environment files...")
    
    # Create .env.example if it doesn't exist
    env_example = project_root / ".env.example"
    if not env_example.exists():
        env_content = """# BU-Processor Environment Configuration
# Copy to .env and adjust values

# Development settings
TESTING=false
BU_LAZY_MODELS=1

# API Configuration
API_HOST=localhost
API_PORT=8000
DEBUG=true

# Cache settings
CACHE_TTL=3600

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/bu_processor.log

# Database (if used)
# DATABASE_URL=sqlite:///bu_processor.db
"""
        with open(env_example, 'w', encoding='utf-8') as f:
            f.write(env_content)
        print_status("Created: .env.example")

def update_gitignore(project_root: Path) -> None:
    """Update .gitignore with comprehensive patterns."""
    print("\n📝 Updating .gitignore...")
    
    gitignore_path = project_root / ".gitignore"
    
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# Project specific
cache/
temp/
logs/
*.bak
.DS_Store
Thumbs.db

# IDE
.vscode/settings.json
.idea/
*.swp
*.swo

# Data files (if sensitive)
data/sensitive/
data/private/
"""
    
    with open(gitignore_path, 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    print_status("Updated: .gitignore")

def organize_workspace():
    """Main function to organize the complete workspace."""
    print("🚀 Organizing BU-Processor Workspace...")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent
    
    # Verify we're in the right place
    if not (project_root / "pyproject.toml").exists():
        print_status("❌ pyproject.toml not found. Are we in the right directory?", success=False)
        return 1
    
    try:
        # 1. Create standard directories
        create_directories(project_root)
        
        # 2. Move documentation
        move_documentation(project_root)
        
        # 3. Organize scripts
        organize_scripts(project_root)
        
        # 4. Clean cache files
        clean_cache_files(project_root)
        
        # 5. Create environment files
        create_environment_files(project_root)
        
        # 6. Update .gitignore
        update_gitignore(project_root)
        
        print("\n✨ Workspace organization complete!")
        print("🎯 Next steps:")
        print("   • Review organized files")
        print("   • Update documentation links")
        print("   • Commit changes")
        
        return 0
        
    except Exception as e:
        print_status(f"Error during organization: {e}", success=False)
        return 1

if __name__ == "__main__":
    exit_code = organize_workspace()
    exit(exit_code)
