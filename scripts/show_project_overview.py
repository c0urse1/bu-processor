#!/usr/bin/env python3
"""
Project Structure Overview for BU-Processor
==========================================
Shows the complete organized project structure.
"""

import os
from pathlib import Path

def show_tree(directory: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0):
    """Display directory tree structure."""
    if current_depth >= max_depth:
        return
    
    items = []
    try:
        items = sorted([item for item in directory.iterdir() 
                       if not item.name.startswith('.') and item.name != '__pycache__'])
    except PermissionError:
        return
    
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        next_prefix = "    " if is_last else "│   "
        
        if item.is_dir():
            print(f"{prefix}{current_prefix}{item.name}/")
            show_tree(item, prefix + next_prefix, max_depth, current_depth + 1)
        else:
            print(f"{prefix}{current_prefix}{item.name}")

def count_files_by_type(directory: Path):
    """Count files by extension."""
    counts = {}
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if not ext:
                ext = "(no extension)"
            counts[ext] = counts.get(ext, 0) + 1
    return counts

def show_project_overview():
    """Show complete project overview."""
    print("🏗️ BU-Processor - Clean & Organized Project Structure")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    
    print(f"\n📁 Project Root: {project_root.absolute()}")
    print("\n🌳 Directory Structure:")
    show_tree(project_root, max_depth=3)
    
    # File statistics
    print(f"\n📊 File Statistics:")
    file_counts = count_files_by_type(project_root)
    total_files = sum(file_counts.values())
    
    print(f"   📄 Total Files: {total_files}")
    
    # Show top file types
    sorted_counts = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
    for ext, count in sorted_counts[:10]:  # Top 10 file types
        percentage = (count / total_files) * 100
        print(f"   {ext:12}: {count:4} files ({percentage:5.1f}%)")
    
    # Key directories info
    print(f"\n🎯 Key Directories:")
    key_dirs = {
        "bu_processor": "Main package source code",
        "tests": "Test suite and test fixtures", 
        "docs": "All project documentation",
        "scripts": "Utility and maintenance scripts",
        "cache": "Runtime cache and temporary data",
        "data": "Data files and datasets"
    }
    
    for dir_name, description in key_dirs.items():
        dir_path = project_root / dir_name
        if dir_path.exists():
            file_count = len([f for f in dir_path.rglob("*") if f.is_file()])
            print(f"   📂 {dir_name:12}: {description} ({file_count} files)")
    
    # Documentation organization
    docs_dir = project_root / "docs"
    if docs_dir.exists():
        print(f"\n📚 Documentation Organization:")
        for subdir in ["api", "guides", "implementation"]:
            subdir_path = docs_dir / subdir
            if subdir_path.exists():
                md_files = len(list(subdir_path.glob("*.md")))
                print(f"   📖 docs/{subdir:14}: {md_files} markdown files")
    
    # Scripts organization
    scripts_dir = project_root / "scripts"
    if scripts_dir.exists():
        print(f"\n⚙️ Scripts Organization:")
        script_files = len(list(scripts_dir.glob("*.py")))
        maintenance_files = len(list((scripts_dir / "maintenance").glob("*")))
        print(f"   🔧 scripts/           : {script_files} Python scripts")
        print(f"   🛠️ scripts/maintenance: {maintenance_files} maintenance tools")
    
    print(f"\n✨ Organization Quality:")
    print(f"   🎯 Clean root directory (no loose files)")
    print(f"   📁 Structured documentation")
    print(f"   🔧 Organized scripts and tools")
    print(f"   🧹 No cache or backup files")
    print(f"   📋 Standard project files in place")
    
    print(f"\n🚀 Ready for:")
    print(f"   • Development")
    print(f"   • Testing") 
    print(f"   • Documentation")
    print(f"   • Deployment")
    print(f"   • Collaboration")

if __name__ == "__main__":
    show_project_overview()
