#!/usr/bin/env python3
"""
Script to fix the syntax error in semantic_chunking_enhancement.py
"""

def fix_syntax_error():
    file_path = "bu_processor/bu_processor/pipeline/semantic_chunking_enhancement.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by lines
    lines = content.split('\n')
    
    # Fix all occurrences of misplaced code after demo_semantic_clustering()
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        new_lines.append(line)
        
        # If we find a demo_semantic_clustering() call
        if line.strip() == 'demo_semantic_clustering()':
            # Skip any misplaced indented code that follows
            i += 1
            while i < len(lines):
                next_line = lines[i]
                # Skip lines that are incorrectly indented method calls or parameters
                if (next_line.strip().startswith('chunk_count=') or
                    next_line.strip().startswith('content_type=') or
                    next_line.strip().startswith('use_context=') or
                    next_line.strip() == ')' or
                    next_line.strip().startswith('# 1. Generiere') or
                    next_line.strip().startswith('embeddings = self._generate') or
                    next_line.strip().startswith('chunks,') or
                    next_line.strip().startswith('use_hierarchical_context,') or
                    next_line.strip().startswith('batch_size') or
                    next_line.strip().startswith('# 2. Führe') or
                    next_line.strip().startswith('cluster_labels = self._perform') or
                    next_line.strip().startswith('# 3. Berechne') or
                    next_line.strip().startswith('similarity_matrix = self._calculate') or
                    next_line.strip().startswith('# 4. Erweitere') or
                    next_line.strip().startswith('enhanced_chunks = self._enrich') or
                    next_line.strip().startswith('# 5. Optimiere') or
                    next_line.strip().startswith('optimized_chunks = self._optimize') or
                    next_line.strip().startswith('processing_time = time.time()') or
                    next_line.strip().startswith('self._total_processing_time') or
                    next_line.strip().startswith('result = SemanticClusterResult') or
                    next_line.strip().startswith('self.logger.info') or
                    next_line.strip().startswith('return result') or
                    (next_line.startswith('            ') and 'chunk_count' in next_line) or
                    (next_line.startswith('        ') and ('embeddings' in next_line or 'cluster_labels' in next_line or 'similarity_matrix' in next_line or 'enhanced_chunks' in next_line))):
                    i += 1
                    continue
                else:
                    # Found a proper line, break and continue
                    break
        else:
            i += 1
    
    # Write the fixed content back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    
    print(f"Fixed syntax error in {file_path}")

if __name__ == "__main__":
    fix_syntax_error()
