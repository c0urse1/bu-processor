#!/usr/bin/env python3
"""
Development Environment Setup Script for BU-Processor
====================================================
Complete setup script for development environment.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"❌ {description} - Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - Error: {e}")
        return False

def setup_development_environment():
    """Set up the complete development environment."""
    print("🚀 Setting up BU-Processor Development Environment")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Steps to execute
    steps = [
        ("python -m pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements-dev.txt", "Installing development dependencies"),
        ("pre-commit install", "Installing pre-commit hooks"),
        ("python scripts/organize_workspace.py", "Organizing workspace"),
        ("python scripts/cleanup.py", "Cleaning up redundant files"),
    ]
    
    success_count = 0
    for command, description in steps:
        if run_command(command, description):
            success_count += 1
        else:
            print(f"⚠️ Warning: {description} failed, but continuing...")
    
    # Check if environment file exists
    env_file = project_root / ".env"
    if not env_file.exists():
        print("\n📝 Creating .env file from template...")
        env_example = project_root / ".env.example"
        if env_example.exists():
            with open(env_example, 'r') as src, open(env_file, 'w') as dst:
                dst.write(src.read())
            print("✅ Created .env file")
        else:
            print("❌ .env.example not found")
    
    # Final verification
    print(f"\n📊 Setup Summary:")
    print(f"   • {success_count}/{len(steps)} steps completed successfully")
    
    if success_count == len(steps):
        print("\n🎉 Development environment setup complete!")
        print("🎯 Ready for:")
        print("   • Code development")
        print("   • Running tests")
        print("   • Git commits with pre-commit hooks")
        print("   • Production deployment")
    else:
        print(f"\n⚠️ Setup completed with {len(steps) - success_count} warnings")
        print("   • Review the failed steps above")
        print("   • Manual intervention may be required")
    
    return success_count == len(steps)

if __name__ == "__main__":
    success = setup_development_environment()
    sys.exit(0 if success else 1)
