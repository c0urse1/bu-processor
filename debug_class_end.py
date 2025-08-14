#!/usr/bin/env python3

with open(r'c:\ml_classifier_poc\bu_processor\bu_processor\pipeline\semantic_chunking_enhancement.py', 'r') as f:
    content = f.read()

lines = content.split('\n')
in_class = False
class_indent = 0

for i, line in enumerate(lines, 1):
    if line.strip().startswith('class SemanticClusteringEnhancer'):
        in_class = True
        class_indent = len(line) - len(line.lstrip())
        print(f'Class starts at line {i}, indent={class_indent}')
    elif in_class and line.strip() and not line.startswith(' ' * (class_indent + 1)):
        print(f'Class ends at line {i-1}')
        print(f'Next line: {repr(line)}')
        break

print(f'Total lines: {len(lines)}')

# Check last few lines
print("\nLast 10 lines:")
for i, line in enumerate(lines[-10:], len(lines)-9):
    print(f"{i:3d}: {repr(line)}")
