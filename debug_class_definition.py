#!/usr/bin/env python3
"""
Debug script to check class definition issues
"""

print("Starting detailed debug...")

# Read the file and check for issues
file_path = r"c:\ml_classifier_poc\bu_processor\bu_processor\pipeline\semantic_chunking_enhancement.py"

print(f"Reading file: {file_path}")

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"File size: {len(content)} characters")

# Look for class definition
import re
class_matches = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
print(f"Found classes: {class_matches}")

# Check for syntax errors by compiling
try:
    compile(content, file_path, 'exec')
    print("✅ File compiles without syntax errors")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
    print(f"Line {e.lineno}: {e.text}")

# Check indentation issues
lines = content.split('\n')
for i, line in enumerate(lines, 1):
    if 'class SemanticClusteringEnhancer' in line:
        print(f"Found class definition at line {i}")
        # Check next few lines for proper indentation
        for j in range(5):
            if i + j < len(lines):
                next_line = lines[i + j]
                if next_line.strip():  # Non-empty line
                    spaces = len(next_line) - len(next_line.lstrip())
                    print(f"Line {i+j+1} indentation: {spaces} spaces - '{next_line[:50]}'")

# Try executing in controlled environment
print("\nTrying controlled execution...")
try:
    namespace = {}
    exec(content, namespace)
    
    print(f"Execution successful. Namespace keys: {list(namespace.keys())}")
    
    if 'SemanticClusteringEnhancer' in namespace:
        print("✅ SemanticClusteringEnhancer found in namespace!")
        cls = namespace['SemanticClusteringEnhancer']
        print(f"Class type: {type(cls)}")
    else:
        print("❌ SemanticClusteringEnhancer not in namespace")
        # Look for any classes
        classes = [k for k, v in namespace.items() if isinstance(v, type)]
        print(f"Classes found: {classes}")
        
except Exception as e:
    print(f"❌ Execution error: {e}")
    import traceback
    traceback.print_exc()
