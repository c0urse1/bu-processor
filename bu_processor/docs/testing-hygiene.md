# Test Organization and Hygiene Guidelines

## 📁 Test Directory Structure

BU-Processor follows strict separation between production code and tests to ensure clean packaging and distribution.

### ✅ Correct Test Placement

```
bu_processor/
├── bu_processor/           # 🚫 NO TEST FILES HERE
│   ├── __init__.py
│   ├── core/
│   ├── pipeline/
│   └── ...
├── tests/                  # ✅ Package-specific tests
│   ├── test_pipeline_components.py
│   ├── test_config.py
│   └── ...
└── scripts/

tests/                      # ✅ Global/integration tests
├── smoke/
├── integration/
└── ...
```

### 🚫 Avoid These Patterns

```bash
# ❌ DON'T place tests inside the main package
bu_processor/bu_processor/test_*.py          # Wrong!
bu_processor/bu_processor/core/test_*.py     # Wrong!
bu_processor/bu_processor/pipeline/test_*.py # Wrong!

# ❌ DON'T import test modules in production code
from .test_utils import helper_function      # Wrong!
import test_helpers                          # Wrong!
```

## 🛡️ Automated Hygiene Checks

### Pre-commit Hooks

Install and configure pre-commit to automatically check test placement:

```bash
pip install pre-commit
pre-commit install
```

This will run checks on every commit to ensure:
- No test files in package directories
- No test imports in production code
- Proper test file naming and placement

### Manual Checks

Run hygiene checks manually:

```bash
# Linux/Mac
./scripts/check-test-hygiene.sh

# Windows
scripts\check-test-hygiene.bat
```

### CI Integration

Add to your CI pipeline:

```yaml
# GitHub Actions example
- name: Check test hygiene
  run: ./scripts/check-test-hygiene.sh
```

## 📦 Distribution Packaging

The `MANIFEST.in` file ensures test files are excluded from distribution packages:

```
# Tests are automatically excluded from wheels/sdist
prune bu_processor/bu_processor/**/test_*.py
recursive-exclude bu_processor/bu_processor test_*.py
```

## 🧪 Test Types and Placement

| Test Type | Location | Purpose |
|-----------|----------|---------|
| Unit Tests | `bu_processor/tests/` | Test individual components |
| Integration Tests | `tests/integration/` | Test component interactions |
| Smoke Tests | `tests/smoke/` | Basic functionality checks |
| End-to-End Tests | `tests/e2e/` | Full workflow testing |

## 🔧 Development Guidelines

1. **Never place test files in `bu_processor/bu_processor/`**
2. **Keep test utilities separate from production code**
3. **Use proper import paths in tests**
4. **Run hygiene checks before commits**
5. **Update this guide when adding new test categories**

## 🐛 Troubleshooting

### Common Issues

**Q: Pre-commit hook fails with "Tests must not live in package dir"**
A: Move the test file from `bu_processor/bu_processor/` to `bu_processor/tests/` or `tests/`

**Q: Import errors when running tests**
A: Ensure you're using absolute imports and running tests from the project root

**Q: Test utilities shared between test files**
A: Create a `conftest.py` file in the appropriate test directory or a separate `test_utils/` package outside the main package

### Getting Help

If you encounter issues with test organization:
1. Check the pre-commit hook output
2. Run the hygiene check script
3. Review this documentation
4. Ask the team for guidance
