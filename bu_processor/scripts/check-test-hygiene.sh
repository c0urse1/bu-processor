#!/bin/bash
# CI Check Script for Test Hygiene
# Usage: ./scripts/check-test-hygiene.sh

set -e

echo "🧪 Checking test file placement hygiene..."

# Check for test files in package directory
TEST_FILES_IN_PACKAGE=$(find bu_processor/bu_processor -name "test_*.py" -o -name "*_test.py" 2>/dev/null || true)

if [ -n "$TEST_FILES_IN_PACKAGE" ]; then
    echo "❌ ERROR: Test files found in package directory:"
    echo "$TEST_FILES_IN_PACKAGE"
    echo ""
    echo "Tests must be placed in:"
    echo "  - tests/ (global tests)"
    echo "  - bu_processor/tests/ (package-specific tests outside main package)"
    echo ""
    echo "NOT in:"
    echo "  - bu_processor/bu_processor/ (main package directory)"
    exit 1
fi

# Check for test imports in production code
echo "🔍 Checking for test imports in production code..."
TEST_IMPORTS=$(find bu_processor/bu_processor -name "*.py" -exec grep -l "import.*test\|from.*test" {} \; 2>/dev/null || true)

if [ -n "$TEST_IMPORTS" ]; then
    echo "⚠️ WARNING: Found potential test imports in production code:"
    echo "$TEST_IMPORTS"
    echo ""
    echo "Production code should not import test modules."
    echo "Consider refactoring shared utilities into a separate module."
fi

# Check that test directories exist and contain tests
echo "📁 Verifying test directory structure..."

if [ ! -d "tests" ]; then
    echo "⚠️ WARNING: Global tests/ directory not found"
fi

if [ ! -d "bu_processor/tests" ]; then
    echo "⚠️ WARNING: Package tests/ directory not found at bu_processor/tests/"
fi

# Count test files in proper locations
GLOBAL_TESTS=$(find tests -name "test_*.py" 2>/dev/null | wc -l || echo 0)
PACKAGE_TESTS=$(find bu_processor/tests -name "test_*.py" 2>/dev/null | wc -l || echo 0)

echo "✅ Test placement hygiene check passed!"
echo "📊 Test file counts:"
echo "  - Global tests: $GLOBAL_TESTS"
echo "  - Package tests: $PACKAGE_TESTS"
echo "  - Total: $((GLOBAL_TESTS + PACKAGE_TESTS))"
