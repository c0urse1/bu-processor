#!/usr/bin/env python3
"""
Workspace Verification Script for BU-Processor
==============================================
Verifies that the workspace is properly organized and configured.
"""

import os
from pathlib import Path
from typing import List, Tuple

def check_directories(project_root: Path) -> Tuple[int, int]:
    """Check for required directories."""
    print("📁 Checking directory structure...")
    
    required_dirs = [
        "bu_processor",
        "tests", 
        "docs",
        "docs/api",
        "docs/guides",
        "docs/implementation",
        "scripts",
        "scripts/maintenance",
        "cache",
        "examples",
        "temp",
        ".github",
        ".github/workflows"
    ]
    
    found = 0
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"   ✅ {dir_name}")
            found += 1
        else:
            print(f"   ❌ {dir_name} (missing)")
    
    return found, len(required_dirs)

def check_files(project_root: Path) -> Tuple[int, int]:
    """Check for required files."""
    print("\n📄 Checking required files...")
    
    required_files = [
        "pyproject.toml",
        ".pre-commit-config.yaml",
        ".gitignore",
        ".env.example",
        "README.md",
        "requirements.txt",
        "requirements-dev.txt"
    ]
    
    found = 0
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists() and file_path.is_file():
            print(f"   ✅ {file_name}")
            found += 1
        else:
            print(f"   ❌ {file_name} (missing)")
    
    return found, len(required_files)

def check_organization(project_root: Path) -> Tuple[int, int]:
    """Check workspace organization."""
    print("\n🏗️ Checking workspace organization...")
    
    checks = [
        ("No loose Python files in root", lambda: len(list(project_root.glob("*.py"))) == 0),
        ("No backup files", lambda: len(list(project_root.rglob("*.bak"))) == 0),
        ("No Python cache in root", lambda: not (project_root / "__pycache__").exists()),
        ("Documentation organized", lambda: len(list((project_root / "docs" / "implementation").glob("*.md"))) > 0),
        ("Scripts organized", lambda: len(list((project_root / "scripts").glob("*.py"))) >= 3),
        ("Maintenance scripts separated", lambda: (project_root / "scripts" / "maintenance").exists()),
    ]
    
    passed = 0
    for description, check_func in checks:
        try:
            if check_func():
                print(f"   ✅ {description}")
                passed += 1
            else:
                print(f"   ❌ {description}")
        except Exception as e:
            print(f"   ❌ {description} (error: {e})")
    
    return passed, len(checks)

def check_cleanliness(project_root: Path) -> Tuple[int, int]:
    """Check for cleanliness issues."""
    print("\n🧹 Checking workspace cleanliness...")
    
    issues = []
    
    # Check for common unwanted files
    unwanted_patterns = [
        "*.pyc",
        "*~",
        "*.swp",
        "*.swo",
        ".DS_Store",
        "Thumbs.db"
    ]
    
    for pattern in unwanted_patterns:
        matches = list(project_root.rglob(pattern))
        if matches:
            issues.append(f"Found {len(matches)} {pattern} files")
    
    # Check for old cache directories
    cache_dirs = [".pytest_cache", "__pycache__"]
    for cache_dir in cache_dirs:
        if (project_root / cache_dir).exists():
            issues.append(f"Found {cache_dir} in root")
    
    # Check for temporary files in root
    temp_files = ["test.bat", "debug_*.py", "temp_*.py"]
    for pattern in temp_files:
        matches = list(project_root.glob(pattern))
        if matches:
            issues.append(f"Found temporary files: {[f.name for f in matches]}")
    
    if not issues:
        print("   ✅ No cleanliness issues found")
        return 1, 1
    else:
        for issue in issues:
            print(f"   ⚠️ {issue}")
        return 0, 1

def verify_workspace():
    """Main verification function."""
    print("🔍 Verifying BU-Processor Workspace")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent
    
    # Verify we're in the right place
    if not (project_root / "pyproject.toml").exists():
        print("❌ pyproject.toml not found. Are we in the right directory?")
        return False
    
    # Run all checks
    dir_score = check_directories(project_root)
    file_score = check_files(project_root)
    org_score = check_organization(project_root)
    clean_score = check_cleanliness(project_root)
    
    # Calculate overall score
    total_found = dir_score[0] + file_score[0] + org_score[0] + clean_score[0]
    total_expected = dir_score[1] + file_score[1] + org_score[1] + clean_score[1]
    
    percentage = (total_found / total_expected) * 100
    
    print(f"\n📊 Verification Summary:")
    print(f"   • Directories: {dir_score[0]}/{dir_score[1]}")
    print(f"   • Files: {file_score[0]}/{file_score[1]}")
    print(f"   • Organization: {org_score[0]}/{org_score[1]}")
    print(f"   • Cleanliness: {clean_score[0]}/{clean_score[1]}")
    print(f"   • Overall Score: {total_found}/{total_expected} ({percentage:.1f}%)")
    
    if percentage >= 90:
        print("\n🎉 Workspace is excellently organized!")
        grade = "A+"
    elif percentage >= 80:
        print("\n👍 Workspace is well organized!")
        grade = "A"
    elif percentage >= 70:
        print("\n👌 Workspace organization is good!")
        grade = "B"
    elif percentage >= 60:
        print("\n⚠️ Workspace needs some improvements!")
        grade = "C"
    else:
        print("\n❌ Workspace needs significant improvements!")
        grade = "D"
    
    print(f"   📈 Grade: {grade}")
    
    return percentage >= 70

if __name__ == "__main__":
    success = verify_workspace()
    exit(0 if success else 1)
