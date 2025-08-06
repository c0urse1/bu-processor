#!/bin/bash
# ============================================================================
# PRE-COMMIT SETUP SCRIPT FOR BU-PROCESSOR
# ============================================================================
# Automatische Installation und Konfiguration der Pre-Commit Hooks

set -e  # Exit on any error

echo "🚀 Setting up Pre-Commit Hooks for BU-Processor..."
echo "================================================="

# Check if we're in the right directory
if [[ ! -f ".pre-commit-config.yaml" ]]; then
    echo "❌ Error: .pre-commit-config.yaml not found. Make sure you're in the project root."
    exit 1
fi

# Install development dependencies (includes pre-commit)
echo "📦 Installing development dependencies..."
pip install -r requirements-dev.txt

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
pre-commit install

# Run pre-commit on all files to check initial setup
echo "✅ Running initial pre-commit check on all files..."
pre-commit run --all-files || {
    echo "⚠️  Some files needed formatting. This is normal for first setup."
    echo "💡 Files have been automatically formatted. Review and commit changes."
}

echo ""
echo "🎉 Pre-commit setup complete!"
echo "================================================="
echo "✓ Pre-commit hooks installed"
echo "✓ Code quality tools configured:"
echo "  - Black (code formatting)"
echo "  - isort (import organization)" 
echo "  - Flake8 (linting)"
echo "  - MyPy (type checking)"
echo "  - General file cleanup hooks"
echo ""
echo "📝 Next steps:"
echo "  - Hooks will run automatically before each commit"
echo "  - To manually run all hooks: pre-commit run --all-files"
echo "  - To skip hooks for a commit: git commit --no-verify"
echo "  - To update hook versions: pre-commit autoupdate"
