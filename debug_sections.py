#!/usr/bin/env python3
"""
Debug by executing sections of the file
"""

file_path = r"c:\ml_classifier_poc\bu_processor\bu_processor\pipeline\semantic_chunking_enhancement.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"File size: {len(content)} characters")

# Split into logical sections and try to execute each
lines = content.split('\n')

# Try to find where the execution might be failing
sections = []
current_section = []
section_name = "header"

for i, line in enumerate(lines):
    if line.strip().startswith('# ==='):
        if current_section:
            sections.append((section_name, current_section))
        section_name = line.strip()
        current_section = []
    else:
        current_section.append(line)

if current_section:
    sections.append((section_name, current_section))

print(f"Found {len(sections)} sections:")
for name, lines in sections:
    print(f"  - {name}: {len(lines)} lines")

# Try to execute each section
namespace = {'__builtins__': __builtins__}

for i, (section_name, section_lines) in enumerate(sections):
    try:
        section_code = '\n'.join(section_lines)
        print(f"\nExecuting section {i+1}: {section_name}")
        
        exec(section_code, namespace)
        
        print(f"✅ Section executed successfully")
        classes_in_namespace = [k for k, v in namespace.items() if isinstance(v, type)]
        print(f"Classes so far: {classes_in_namespace}")
        
    except Exception as e:
        print(f"❌ Error in section {section_name}: {e}")
        import traceback
        traceback.print_exc()
        break

print(f"\nFinal namespace classes: {[k for k, v in namespace.items() if isinstance(v, type)]}")
if 'SemanticClusteringEnhancer' in namespace:
    print("✅ SemanticClusteringEnhancer found in final namespace!")
else:
    print("❌ SemanticClusteringEnhancer not found")
