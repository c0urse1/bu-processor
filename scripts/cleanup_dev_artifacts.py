#!/usr/bin/env python3
"""
Development Artifacts Cleanup Script for BU-Processor
====================================================
Removes all development test files and artifacts that were created during
iterative development and testing.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List
import shutil

def print_status(message: str, success: bool = True, dry_run: bool = False):
    """Print colored status message."""
    if dry_run:
        emoji = "🔍" if success else "⚠️"
        prefix = "[DRY RUN] "
    else:
        emoji = "✅" if success else "❌"
        prefix = ""
    print(f"{emoji} {prefix}{message}")

def safe_remove_file(file_path: Path, dry_run: bool = False) -> bool:
    """Remove a file safely and return True on success."""
    try:
        if file_path.exists() and file_path.is_file():
            if dry_run:
                print_status(f"Would remove: {file_path.name}", dry_run=True)
                return True
            else:
                file_path.unlink()
                print_status(f"Removed: {file_path.name}")
                return True
        return False
    except Exception as e:
        print_status(f"Error removing {file_path.name}: {e}", success=False, dry_run=dry_run)
        return False

def cleanup_development_artifacts(dry_run: bool = False):
    """Main cleanup function for development artifacts."""
    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"🧹 BU-Processor Development Artifacts Cleanup ({mode})")
    print("=" * 60)

    # Find project root
    project_root = Path(__file__).parent.parent
    bu_processor_dir = project_root / "bu_processor"
    
    if not bu_processor_dir.exists():
        print_status("bu_processor directory not found!", success=False, dry_run=dry_run)
        return 1

    # Counters for summary
    counts = {
        "test_files": 0,
        "verify_files": 0,
        "debug_files": 0,
        "development_files": 0,
        "temp_files": 0
    }

    # 1. Remove test_*.py files from bu_processor directory
    print("\n🧪 Removing development test files:")
    test_files = list(bu_processor_dir.glob("test_*.py"))
    for file_path in test_files:
        if safe_remove_file(file_path, dry_run):
            counts["test_files"] += 1
    print(f"   → {counts['test_files']} test files {'would be removed' if dry_run else 'removed'}")

    # 2. Remove verify_*.py files
    print("\n✅ Removing verification scripts:")
    verify_files = list(bu_processor_dir.glob("verify_*.py"))
    for file_path in verify_files:
        if safe_remove_file(file_path, dry_run):
            counts["verify_files"] += 1
    print(f"   → {counts['verify_files']} verify scripts {'would be removed' if dry_run else 'removed'}")

    # 3. Remove debug and development files
    print("\n🐛 Removing debug and development files:")
    development_patterns = [
        "debug_*.py",
        "simple_*.py", 
        "direct_*.py",
        "quick_*.py",
        "minimal_*.py",
        "temp_*.py",
        "backup_*.py",
        "fix_*.py",
        "final_*.py"
    ]
    
    for pattern in development_patterns:
        files = list(bu_processor_dir.glob(pattern))
        for file_path in files:
            if safe_remove_file(file_path, dry_run):
                counts["development_files"] += 1
    print(f"   → {counts['development_files']} development files {'would be removed' if dry_run else 'removed'}")

    # 4. Remove other temporary files
    print("\n🗑️ Removing temporary files:")
    temp_patterns = [
        "create_training_csvs_quickfix.py",
        "validate_fixes.py",
        "*.bak",
        "*.tmp",
        "*_old.py",
        "*_backup.py"
    ]
    
    for pattern in temp_patterns:
        files = list(bu_processor_dir.glob(pattern))
        for file_path in files:
            if safe_remove_file(file_path, dry_run):
                counts["temp_files"] += 1
    print(f"   → {counts['temp_files']} temporary files {'would be removed' if dry_run else 'removed'}")

    # 5. Check for any remaining loose Python files that aren't core modules
    print("\n📋 Checking for remaining loose files:")
    
    # Core files that should remain in bu_processor
    core_files = {
        "__init__.py",
        "start_api.py",
        "models.py",
        "processor.py", 
        "api.py",
        "cli.py",
        "config.py",
        "utils.py"
    }
    
    all_py_files = list(bu_processor_dir.glob("*.py"))
    loose_files = [f for f in all_py_files if f.name not in core_files]
    
    if loose_files:
        print("   ⚠️ Found additional loose files (review manually):")
        for file_path in loose_files:
            print(f"      • {file_path.name}")
    else:
        print("   ✅ No additional loose files found")

    # Summary
    total_removed = sum(counts.values())
    print(f"\n✨ Cleanup {'simulation' if dry_run else 'completed'}!")
    print("📊 Summary:")
    verb = "would be removed" if dry_run else "removed"
    print(f"   • {counts['test_files']} test files {verb}")
    print(f"   • {counts['verify_files']} verify scripts {verb}")
    print(f"   • {counts['development_files']} development files {verb}")
    print(f"   • {counts['temp_files']} temporary files {verb}")
    print(f"   • Total: {total_removed} files {verb}")

    if not dry_run and total_removed > 0:
        print("\n🎯 Repository is now cleaner!")
        print("   • Only core module files remain in bu_processor/")
        print("   • All tests are properly organized in tests/")
        print("   • Development artifacts removed")
    elif dry_run and total_removed > 0:
        print(f"\n🎯 Run without --dry-run to remove {total_removed} files")
    else:
        print("\n✅ Repository is already clean!")
        
    return 0

def main():
    """CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="BU-Processor Development Artifacts Cleanup")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be removed without actually removing files")
    
    args = parser.parse_args()
    
    try:
        exit_code = cleanup_development_artifacts(dry_run=args.dry_run)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n❌ Cleanup aborted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
