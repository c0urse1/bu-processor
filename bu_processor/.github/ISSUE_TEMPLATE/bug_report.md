---
name: 🐛 Bug Report
about: Report a bug to help us improve BU-Processor
title: '[BUG] '
labels: ['bug', 'needs-triage']
assignees: ''
---

## 🐛 Bug Description
<!-- Clear and concise description of what the bug is -->

## 🔄 Steps to Reproduce
<!-- Exact steps to reproduce the behavior -->
1. Go to '...'
2. Click on '...'
3. Run command '...'
4. See error

## 🎯 Expected Behavior
<!-- Clear description of what you expected to happen -->

## 💥 Actual Behavior
<!-- What actually happened? Include full error messages -->

```
Error message here (if any)
```

## 🖥️ Environment
<!-- Please complete the following information -->
- **OS**: [e.g. Windows 11, Ubuntu 22.04, macOS 13.0]
- **Python Version**: [e.g. 3.9.7]
- **BU-Processor Version**: [e.g. 3.0.0]
- **GPU**: [e.g. NVIDIA RTX 3080, None, AMD RX 6800]
- **Available RAM**: [e.g. 16GB]

## 📋 Configuration
<!-- Your .env configuration (remove sensitive data!) -->
```bash
BU_PROCESSOR_ENVIRONMENT=development
BU_PROCESSOR_ML_MODEL__MODEL_PATH=bert-base-german-cased
BU_PROCESSOR_PDF_PROCESSING__MAX_PDF_SIZE_MB=50
# ... other relevant config
```

## 📄 Sample Data
<!-- If applicable, describe the input that caused the issue -->
- **PDF Size**: [e.g. 2.5MB, 50 pages]
- **Content Type**: [e.g. Legal text, Technical manual, Mixed]
- **Language**: [e.g. German, English, Mixed]
- **Special Characteristics**: [e.g. scanned document, many tables, handwritten notes]

<!-- If possible, attach a minimal sample that reproduces the issue -->

## 📊 Logs & Output
<!-- Include relevant log output, stack traces, console output -->
<details>
<summary>Log Output</summary>

```
Paste log output here
```
</details>

## 🔍 Additional Context
<!-- Add any other context about the problem here -->
- When did this start happening?
- Does it happen consistently or intermittently?
- Any workarounds you've found?
- Related issues or discussions?

## ✅ Pre-submission Checklist
- [ ] I have searched existing issues to avoid duplicates
- [ ] I have provided all requested information
- [ ] I have tested with the latest version
- [ ] I have included relevant logs and error messages
- [ ] I have removed any sensitive information from the report
