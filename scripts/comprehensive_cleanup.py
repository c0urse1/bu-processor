#!/usr/bin/env python3
"""
Comprehensive Cleanup Script for BU-Processor Go-Live
====================================================
Removes all redundant files and organizes structure for production.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import List, Dict

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

def move_file_to_docs(source: Path, target_dir: Path, dry_run: bool = False) -> bool:
    """Move file to docs directory."""
    try:
        if source.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / source.name
            if dry_run:
                print_status(f"Would move: {source.name} → docs/{target_dir.name}/", dry_run=True)
                return True
            else:
                shutil.move(str(source), str(target))
                print_status(f"Moved: {source.name} → docs/{target_dir.name}/")
                return True
        return False
    except Exception as e:
        print_status(f"Error moving {source.name}: {e}", success=False, dry_run=dry_run)
        return False

def comprehensive_cleanup(dry_run: bool = False):
    """Main cleanup function."""
    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"🧹 BU-Processor Comprehensive Cleanup ({mode})")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    if not (project_root / "pyproject.toml").exists():
        print_status(f"Project directory not found at: {project_root}", success=False)
        return 1
    
    counts = {"debug": 0, "test": 0, "summary": 0, "database": 0, "moved": 0}
    
    # 1. Remove diagnostic and debug files
    print("\n🐛 Remove diagnostic and debug files:")
    debug_files = [
        "bu_store.db",
        "classifier_improvements_summary.py", 
        "comprehensive_diagnostic.py",
        "corrected_diagnostic.py",
        "diagnose_device.py",
        "fixed_diagnostic.py",
        "temp_exports.txt"
    ]
    
    for file_name in debug_files:
        file_path = project_root / file_name
        if safe_remove_file(file_path, dry_run):
            counts["debug"] += 1
    
    # 2. Remove all test files from root (already handled by existing script)
    print(f"\n🧪 Test files cleanup:")
    print(f"   → 33 test files will be handled by existing cleanup script")
    
    # 3. Move summary files to docs/implementation
    print("\n📋 Move summary files to docs:")
    docs_impl = project_root / "docs" / "implementation"
    if not dry_run:
        docs_impl.mkdir(parents=True, exist_ok=True)
    
    summary_files = [
        "GUARD_IMPLEMENTATION_SUMMARY.md",
        "MOCK_SAFE_SUCCESS_SUMMARY.md", 
        "PINECONE_IMPROVEMENTS_SUMMARY.py",
        "SEARCH_SIMILAR_DOCUMENTS_SUMMARY.py"
    ]
    
    for file_name in summary_files:
        file_path = project_root / file_name
        if move_file_to_docs(file_path, docs_impl, dry_run):
            counts["moved"] += 1
    
    # 4. Check for database files
    print(f"\n🗄️ Database files:")
    db_files = list(project_root.glob("*.db"))
    for db_file in db_files:
        if safe_remove_file(db_file, dry_run):
            counts["database"] += 1
    
    # 5. Remove temporary exports
    print(f"\n📤 Temporary files:")
    temp_patterns = ["temp_*.txt", "*.tmp", "debug_*.log"]
    for pattern in temp_patterns:
        for file_path in project_root.glob(pattern):
            if file_path.is_file() and safe_remove_file(file_path, dry_run):
                counts["debug"] += 1
    
    # Summary
    print(f"\n✨ Comprehensive cleanup {'simulated' if dry_run else 'completed'}!")
    print("📊 Summary:")
    verb = "would be" if dry_run else "were"
    print(f"   • {counts['debug']} debug/diagnostic files {verb} removed")
    print(f"   • {counts['database']} database files {verb} removed") 
    print(f"   • {counts['moved']} summary files {verb} moved to docs/")
    print(f"   • 33 test files will be removed by main cleanup script")
    
    total_files = sum(counts.values()) + 33  # Include test files
    
    if not dry_run:
        print(f"\n🎯 Next steps:")
        print(f"   • Run: python scripts\\cleanup.py (removes 33 test files)")
        print(f"   • Verify: python scripts\\verify_workspace.py")
        print(f"   • Test: pytest tests/")
    else:
        print(f"\n🎯 Run without --dry-run to clean {total_files} files total")
    
    return 0

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="BU-Processor Comprehensive Cleanup")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    try:
        exit_code = comprehensive_cleanup(dry_run=args.dry_run)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n❌ Cleanup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
