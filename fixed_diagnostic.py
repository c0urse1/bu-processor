#!/usr/bin/env python3
"""
🔍 BU-PROCESSOR PROJECT COMPREHENSIVE DIAGNOSTIC (FIXED)
Corrected to use actual available modules and classes
"""

import sys
import os
from pathlib import Path
import subprocess
import importlib.util
import time

# CORRECTED Configuration - Understanding the nested structure
WORKSPACE_ROOT = Path("C:/ml_classifier_poc")
BU_PROCESSOR_ROOT = WORKSPACE_ROOT / "bu_processor"  # Git repo root
BU_PROCESSOR_PACKAGE = BU_PROCESSOR_ROOT / "bu_processor"  # Actual Python package

def print_header(title, level=1):
    """Print formatted section headers"""
    if level == 1:
        print(f"\n{'='*80}")
        print(f"🔍 {title}")
        print('='*80)
    elif level == 2:
        print(f"\n{'-'*60}")
        print(f"📋 {title}")
        print('-'*60)
    else:
        print(f"\n📦 {title}")

def check_structure():
    """Check project directory structure"""
    print_header("PROJECT STRUCTURE ASSESSMENT", 2)
    
    expected_structure = {
        # Main workspace
        "workspace root": WORKSPACE_ROOT,
        "bu_processor repo": BU_PROCESSOR_ROOT,
        "bu_processor package": BU_PROCESSOR_PACKAGE,
        
        # Package structure
        "package __init__.py": BU_PROCESSOR_PACKAGE / "__init__.py",
        "config.py": BU_PROCESSOR_PACKAGE / "config.py",
        "factories.py": BU_PROCESSOR_PACKAGE / "factories.py",
        "ports.py": BU_PROCESSOR_PACKAGE / "ports.py",
        
        # Key modules
        "telemetry/": BU_PROCESSOR_PACKAGE / "telemetry",
        "eval/": BU_PROCESSOR_PACKAGE / "eval", 
        "answering/": BU_PROCESSOR_PACKAGE / "answering",
        "retrieval/": BU_PROCESSOR_PACKAGE / "retrieval",
        "core/": BU_PROCESSOR_PACKAGE / "core",
        
        # Config files
        ".env": BU_PROCESSOR_ROOT / ".env",
        "pyproject.toml": BU_PROCESSOR_ROOT / "pyproject.toml",
        "requirements.txt": BU_PROCESSOR_ROOT / "requirements.txt",
        
        # Test structure
        "tests/": BU_PROCESSOR_ROOT / "tests",
        "tools/": BU_PROCESSOR_ROOT / "tools",
        
        # Documentation
        "README.md": BU_PROCESSOR_ROOT / "README.md",
        "docs/": BU_PROCESSOR_ROOT / "docs",
    }
    
    structure_score = 0
    total_checks = len(expected_structure)
    
    for name, path in expected_structure.items():
        if path.exists():
            size_info = ""
            if path.is_file():
                size_info = f"({path.stat().st_size} bytes)"
            elif path.is_dir():
                try:
                    items = list(path.iterdir())
                    size_info = f"({len(items)} items)"
                except PermissionError:
                    size_info = "(permission denied)"
            
            print(f"✅ {name:<35} {size_info}")
            structure_score += 1
        else:
            print(f"❌ {name:<35} MISSING")
    
    percentage = (structure_score / total_checks) * 100
    print(f"\n📊 Structure Score: {structure_score}/{total_checks} ({percentage:.1f}%)")
    return percentage

def check_imports():
    """Test critical import paths"""
    print_header("IMPORT DEPENDENCY CHECK", 2)
    
    # Add the bu_processor package to Python path
    sys.path.insert(0, str(BU_PROCESSOR_ROOT))
    
    critical_imports = [
        "bu_processor",
        "bu_processor.config", 
        "bu_processor.factories",
        "bu_processor.ports",
        "bu_processor.telemetry.trace",
        "bu_processor.telemetry.wrap",
        "bu_processor.eval.metrics",
        "bu_processor.eval.harness",
        "bu_processor.answering.synthesize",
        "bu_processor.retrieval.models",
    ]
    
    import_score = 0
    for module_name in critical_imports:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name}")
            import_score += 1
        except ImportError as e:
            print(f"❌ {module_name} - {e}")
        except Exception as e:
            print(f"⚠️  {module_name} - {type(e).__name__}: {e}")
    
    percentage = (import_score / len(critical_imports)) * 100
    print(f"\n📊 Import Score: {import_score}/{len(critical_imports)} ({percentage:.1f}%)")
    return percentage

def check_functionality():
    """Test core functionality"""
    print_header("CORE FUNCTIONALITY TEST", 2)
    
    sys.path.insert(0, str(BU_PROCESSOR_ROOT))
    
    tests = []
    
    # Test 1: Configuration creation
    try:
        from bu_processor.config import Settings
        settings = Settings()
        tests.append(("Configuration creation", True, f"Embeddings: {settings.EMBEDDINGS_BACKEND}"))
    except Exception as e:
        tests.append(("Configuration creation", False, str(e)))
    
    # Test 2: Factory creation
    try:
        from bu_processor.factories import make_hybrid_retriever, make_answerer
        retriever = make_hybrid_retriever()
        answerer = make_answerer()
        tests.append(("Factory creation", True, "Retriever and answerer created"))
    except Exception as e:
        tests.append(("Factory creation", False, str(e)))
    
    # Test 3: Trace logging (corrected)
    try:
        from bu_processor.telemetry.trace import Trace
        trace = Trace()
        trace.event("test_event", data="test")
        with trace.stage("test_stage"):
            pass
        tests.append(("Trace logging", True, f"Trace with {len(trace.events)} events"))
    except Exception as e:
        tests.append(("Trace logging", False, str(e)))
    
    # Test 4: Evaluation metrics (simplified)
    try:
        from bu_processor.eval.metrics import hit_at_k, reciprocal_rank
        
        hit_score = hit_at_k(["doc1", "doc2"], ["doc1"], k=1)
        mrr_score = reciprocal_rank(["doc1", "doc2"], ["doc1"])
        tests.append(("Evaluation metrics", True, f"Hit@1: {hit_score:.2f}, MRR: {mrr_score:.2f}"))
    except Exception as e:
        tests.append(("Evaluation metrics", False, str(e)))
    
    # Test 5: Complete pipeline (corrected)
    try:
        from bu_processor.factories import make_hybrid_retriever, make_answerer
        from bu_processor.answering.synthesize import synthesize_answer
        
        r = make_hybrid_retriever()
        a = make_answerer()
        hits = r.retrieve('test query', final_top_k=3)
        result = synthesize_answer(query='test', hits=hits, answerer=a, token_budget=500)
        tests.append(("Complete pipeline", True, f"Answer: {len(result.text)} chars, {len(result.citations)} citations"))
    except Exception as e:
        tests.append(("Complete pipeline", False, str(e)))
    
    # Display results
    passed = 0
    for name, success, details in tests:
        if success:
            print(f"✅ {name:<25} - {details}")
            passed += 1
        else:
            print(f"❌ {name:<25} - {details}")
    
    percentage = (passed / len(tests)) * 100
    print(f"\n📊 Functionality Score: {passed}/{len(tests)} ({percentage:.1f}%)")
    return percentage

def check_tests():
    """Run test suites"""
    print_header("TEST SUITE EXECUTION", 2)
    
    os.chdir(BU_PROCESSOR_ROOT)
    
    test_files = [
        "test_answer_synthesis_simple.py",
        "tests/test_telemetry.py",
        "tests/test_eval_metrics.py", 
        "tests/test_eval_harness.py",
    ]
    
    passed_tests = 0
    total_tests = len(test_files)
    
    for test_file in test_files:
        if (BU_PROCESSOR_ROOT / test_file).exists():
            try:
                result = subprocess.run(
                    [sys.executable, test_file], 
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                if result.returncode == 0:
                    print(f"✅ {test_file}")
                    passed_tests += 1
                else:
                    print(f"❌ {test_file} - Exit code {result.returncode}")
                    if result.stderr:
                        print(f"   Error: {result.stderr[:100]}...")
            except subprocess.TimeoutExpired:
                print(f"⏱️  {test_file} - Timeout")
            except Exception as e:
                print(f"💥 {test_file} - {e}")
        else:
            print(f"❓ {test_file} - File not found")
    
    percentage = (passed_tests / total_tests) * 100
    print(f"\n📊 Test Score: {passed_tests}/{total_tests} ({percentage:.1f}%)")
    return percentage

def check_demos():
    """Test integration demos"""
    print_header("INTEGRATION DEMO CHECK", 2)
    
    os.chdir(BU_PROCESSOR_ROOT)
    
    demos = [
        "final_system_demo.py",
        "demo_complete_system.py", 
        "tools/run_eval_and_gate.py",
    ]
    
    passed_demos = 0
    total_demos = len(demos)
    
    for demo in demos:
        if (BU_PROCESSOR_ROOT / demo).exists():
            try:
                # Just check if the file can be imported/parsed
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile", demo], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                if result.returncode == 0:
                    print(f"✅ {demo} - Syntax OK")
                    passed_demos += 1
                else:
                    print(f"❌ {demo} - Syntax error")
            except Exception as e:
                print(f"💥 {demo} - {e}")
        else:
            print(f"❓ {demo} - File not found")
    
    percentage = (passed_demos / total_demos) * 100
    print(f"\n📊 Demo Score: {passed_demos}/{total_demos} ({percentage:.1f}%)")
    return percentage

def main():
    """Run comprehensive diagnostic"""
    print_header("BU-PROCESSOR PROJECT HEALTH DIAGNOSTIC (FIXED)")
    
    start_time = time.time()
    
    # Run all checks
    structure_score = check_structure()
    import_score = check_imports() 
    functionality_score = check_functionality()
    test_score = check_tests()
    demo_score = check_demos()
    
    # Calculate overall health
    overall_score = (
        structure_score * 0.2 +  # 20% weight
        import_score * 0.3 +     # 30% weight  
        functionality_score * 0.3 + # 30% weight
        test_score * 0.1 +       # 10% weight
        demo_score * 0.1         # 10% weight
    )
    
    # Final report
    print_header("FINAL HEALTH REPORT")
    print(f"📊 Structure:     {structure_score:6.1f}%")
    print(f"📦 Imports:       {import_score:6.1f}%") 
    print(f"⚙️  Functionality: {functionality_score:6.1f}%")
    print(f"🧪 Tests:         {test_score:6.1f}%")
    print(f"🎭 Demos:         {demo_score:6.1f}%")
    print(f"{'='*40}")
    print(f"🎯 OVERALL HEALTH: {overall_score:6.1f}%")
    
    elapsed = time.time() - start_time
    print(f"⏱️  Diagnostic completed in {elapsed:.1f}s")
    
    # Health interpretation
    if overall_score >= 90:
        print("🟢 EXCELLENT: Production ready!")
    elif overall_score >= 75:
        print("🟡 GOOD: Minor issues to address")
    elif overall_score >= 50:
        print("🟠 FAIR: Several components need attention")
    else:
        print("🔴 POOR: Major issues require immediate attention")

if __name__ == "__main__":
    main()
