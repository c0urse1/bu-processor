#!/usr/bin/env python3
"""
Move Test Files from Package Directory
=====================================
Moves all test files from bu_processor/ to tests/ directory.
"""

import os
import shutil
from pathlib import Path

def move_tests_from_package():
    """Move test files from package to tests directory."""
    print("🔄 Moving test files from package directory...")
    
    project_root = Path(__file__).parent.parent
    bu_processor_dir = project_root / "bu_processor"
    tests_dir = project_root / "tests"
    
    # Ensure tests directory exists
    tests_dir.mkdir(exist_ok=True)
    
    # Find all test files in bu_processor
    test_files = []
    for pattern in ["test_*.py", "*_test.py", "debug_*.py"]:
        test_files.extend(bu_processor_dir.glob(pattern))
    
    moved_count = 0
    for test_file in test_files:
        if test_file.is_file():
            target = tests_dir / test_file.name
            try:
                # Avoid overwriting existing files
                if target.exists():
                    target = tests_dir / f"moved_{test_file.name}"
                
                shutil.move(str(test_file), str(target))
                print(f"✅ Moved: {test_file.name} → tests/")
                moved_count += 1
            except Exception as e:
                print(f"❌ Error moving {test_file.name}: {e}")
    
    print(f"\n✨ Moved {moved_count} test files to tests/ directory")
    return moved_count

if __name__ == "__main__":
    move_tests_from_package()
