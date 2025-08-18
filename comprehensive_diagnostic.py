#!/usr/bin/env python3
"""
🔍 BU-PROCESSOR PROJECT COMPREHENSIVE DIAGNOSTIC
Full assessment of project health, test coverage, and component status
"""

import sys
import os
from pathlib import Path
import subprocess
import importlib.util
import time

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
        print(f"\n🔸 {title}")

def run_command(command, timeout=30, show_output=True):
    """Run a command and return success status with output"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd=os.getcwd()
        )
        
        success = result.returncode == 0
        output = result.stdout.strip() if result.stdout else ""
        error = result.stderr.strip() if result.stderr else ""
        
        if show_output:
            if success:
                print(f"✅ SUCCESS")
                if output:
                    # Show last few lines of output
                    lines = output.split('\n')[-5:]
                    for line in lines:
                        if line.strip():
                            print(f"   {line}")
            else:
                print(f"❌ FAILED (exit code: {result.returncode})")
                if error:
                    lines = error.split('\n')[:3]  # Show first few error lines
                    for line in lines:
                        if line.strip():
                            print(f"   ERROR: {line}")
                elif output:
                    lines = output.split('\n')[-3:]
                    for line in lines:
                        if line.strip():
                            print(f"   OUTPUT: {line}")
        
        return success, output, error
    
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT after {timeout}s")
        return False, "", "Timeout"
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")
        return False, "", str(e)

def check_file_structure():
    """Check critical project files and directories"""
    print_header("PROJECT STRUCTURE", 2)
    
    critical_paths = [
        # Core project structure
        ("bu_processor/", "Main package directory"),
        ("bu_processor/__init__.py", "Package init"),
        ("bu_processor/bu_processor/", "Core module"),
        ("bu_processor/bu_processor/__init__.py", "Core init"),
        
        # Configuration
        ("bu_processor/config.py", "Configuration"),
        ("bu_processor/.env", "Environment config"),
        ("bu_processor/pyproject.toml", "Project config"),
        
        # Test structure
        ("tests/", "Test directory"),
        ("bu_processor/tests/", "Internal tests"),
        
        # Documentation
        ("docs/", "Documentation"),
        ("README.md", "Main readme"),
        
        # Virtual environment
        ("bu_processor/venv/", "Virtual environment"),
        
        # Recent implementations
        ("bu_processor/bu_processor/telemetry/", "Telemetry system"),
        ("bu_processor/bu_processor/eval/", "Evaluation system"),
        ("bu_processor/bu_processor/answering/", "Answer synthesis"),
        ("bu_processor/bu_processor/retrieval/", "Retrieval system"),
    ]
    
    structure_ok = True
    for path, description in critical_paths:
        full_path = Path(path)
        if full_path.exists():
            if full_path.is_dir():
                file_count = len(list(full_path.glob("*")))
                print(f"✅ {path:<40} ({file_count} items)")
            else:
                size = full_path.stat().st_size
                print(f"✅ {path:<40} ({size} bytes)")
        else:
            print(f"❌ {path:<40} MISSING")
            structure_ok = False
    
    return structure_ok

def check_python_environment():
    """Check Python environment and key dependencies"""
    print_header("PYTHON ENVIRONMENT", 2)
    
    # Python version
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📦 Python path: {sys.executable}")
    
    # Key imports
    key_imports = [
        ("bu_processor", "Main package"),
        ("pytest", "Testing framework"),
        ("numpy", "Numerical computing"),
        ("sentence_transformers", "Embeddings"),
        ("faiss", "Vector index"),
        ("pydantic", "Data validation"),
        ("sqlalchemy", "Database ORM"),
    ]
    
    import_results = {}
    for module, description in key_imports:
        try:
            spec = importlib.util.find_spec(module)
            if spec is not None:
                print(f"✅ {module:<20} - {description}")
                import_results[module] = True
            else:
                print(f"❌ {module:<20} - {description} (not found)")
                import_results[module] = False
        except Exception as e:
            print(f"❌ {module:<20} - {description} (error: {e})")
            import_results[module] = False
    
    return import_results

def run_core_imports():
    """Test core package imports"""
    print_header("CORE IMPORTS", 2)
    
    core_imports = [
        "bu_processor",
        "bu_processor.config",
        "bu_processor.factories",
        "bu_processor.telemetry.trace",
        "bu_processor.eval.metrics",
        "bu_processor.answering.synthesize",
        "bu_processor.retrieval.hybrid",
        "bu_processor.embeddings.testing_backend",
    ]
    
    import_success = 0
    total_imports = len(core_imports)
    
    for module in core_imports:
        print(f"Testing: {module}")
        success, output, error = run_command(f"python -c \"import {module}; print('✓ {module} imported successfully')\"", timeout=10)
        if success:
            import_success += 1
        print()
    
    print(f"📊 Import Results: {import_success}/{total_imports} successful")
    return import_success, total_imports

def run_basic_functionality():
    """Test basic functionality"""
    print_header("BASIC FUNCTIONALITY", 2)
    
    functionality_tests = [
        ("Config Loading", "python -c \"from bu_processor.config import settings; print(f'Backend: {settings.EMBEDDINGS_BACKEND}')\""),
        ("Factory Creation", "python -c \"from bu_processor.factories import make_embedder; embedder = make_embedder(); print('Factory works')\""),
        ("Trace Creation", "python -c \"from bu_processor.telemetry.trace import Trace; t = Trace(); t.event('test'); print(f'Trace ID: {t.trace_id}')\""),
        ("Metrics Calculation", "python -c \"from bu_processor.eval.metrics import hit_at_k; result = hit_at_k(['a','b'], ['b'], 2); print(f'Hit@2: {result}')\""),
    ]
    
    func_success = 0
    total_funcs = len(functionality_tests)
    
    for name, command in functionality_tests:
        print(f"Testing: {name}")
        success, output, error = run_command(command, timeout=15)
        if success:
            func_success += 1
        print()
    
    print(f"📊 Functionality Results: {func_success}/{total_funcs} successful")
    return func_success, total_funcs

def run_test_suites():
    """Run various test suites"""
    print_header("TEST SUITES", 2)
    
    test_commands = [
        ("Telemetry Tests", "python -c \"from tests.test_telemetry import *; test_trace_basic(); test_serialize_hits(); print('Telemetry tests passed')\""),
        ("Evaluation Metrics", "python -c \"from tests.test_eval_metrics import *; test_hit_at_k(); test_citation_accuracy(); print('Eval metrics tests passed')\""),
        ("Answer Synthesis", "python test_answer_synthesis_simple.py"),
        ("Simple Pipeline", "python -c \"from bu_processor.factories import make_embedder; e = make_embedder(); print('Pipeline components work')\""),
    ]
    
    test_success = 0
    total_tests = len(test_commands)
    
    for name, command in test_commands:
        print(f"Running: {name}")
        success, output, error = run_command(command, timeout=30)
        if success:
            test_success += 1
        print()
    
    # Try pytest if available
    print("Running pytest (if available)...")
    pytest_success, pytest_output, pytest_error = run_command("pytest tests/ --tb=line -q --maxfail=5", timeout=60)
    
    print(f"📊 Test Results: {test_success}/{total_tests} manual tests successful")
    if pytest_success:
        print("✅ Pytest suite also passed")
    else:
        print("❌ Pytest suite had issues")
    
    return test_success, total_tests, pytest_success

def run_integration_demos():
    """Run integration demonstrations"""
    print_header("INTEGRATION DEMOS", 2)
    
    demo_commands = [
        ("Trace CLI", "python bu_processor/cli_run_with_trace.py"),
        ("Evaluation Demo", "python tools/run_eval_and_gate.py"),
        ("System Demo", "python final_system_demo.py"),
    ]
    
    demo_success = 0
    total_demos = len(demo_commands)
    
    for name, command in demo_commands:
        print(f"Running: {name}")
        success, output, error = run_command(command, timeout=45)
        if success:
            demo_success += 1
        print()
    
    print(f"📊 Demo Results: {demo_success}/{total_demos} successful")
    return demo_success, total_demos

def generate_summary(results):
    """Generate comprehensive summary"""
    print_header("COMPREHENSIVE SUMMARY", 1)
    
    structure_ok, import_results, (import_success, total_imports), (func_success, total_funcs), (test_success, total_tests, pytest_success), (demo_success, total_demos) = results
    
    # Overall health calculation
    structure_score = 100 if structure_ok else 0
    import_score = (sum(import_results.values()) / len(import_results)) * 100
    func_score = (func_success / total_funcs) * 100
    test_score = (test_success / total_tests) * 100
    demo_score = (demo_success / total_demos) * 100
    
    overall_score = (structure_score + import_score + func_score + test_score + demo_score) / 5
    
    print(f"📊 PROJECT HEALTH REPORT")
    print(f"{'='*50}")
    print(f"📁 Project Structure:     {'✅' if structure_ok else '❌'} ({structure_score:.0f}%)")
    print(f"📦 Import Dependencies:   {sum(import_results.values())}/{len(import_results)} ({import_score:.0f}%)")
    print(f"⚙️  Core Functionality:    {func_success}/{total_funcs} ({func_score:.0f}%)")
    print(f"🧪 Test Suites:          {test_success}/{total_tests} ({test_score:.0f}%)")
    print(f"🎯 Integration Demos:     {demo_success}/{total_demos} ({demo_score:.0f}%)")
    print(f"🏆 OVERALL HEALTH:        {overall_score:.1f}%")
    
    if overall_score >= 80:
        status_emoji = "🟢"
        status_text = "EXCELLENT - Production ready"
    elif overall_score >= 60:
        status_emoji = "🟡"
        status_text = "GOOD - Minor issues to address"
    elif overall_score >= 40:
        status_emoji = "🟠"
        status_text = "FAIR - Significant issues need attention"
    else:
        status_emoji = "🔴"
        status_text = "NEEDS WORK - Major issues present"
    
    print(f"\n{status_emoji} STATUS: {status_text}")
    
    # Recommendations
    print(f"\n🔧 RECOMMENDATIONS:")
    if not structure_ok:
        print("   - Fix missing project structure files")
    if import_score < 100:
        print("   - Install missing dependencies")
    if func_score < 80:
        print("   - Debug core functionality issues")
    if test_score < 70:
        print("   - Fix failing test suites")
    if demo_score < 70:
        print("   - Address integration demo issues")
    
    print(f"\n📈 NEXT STEPS:")
    print(f"   1. Address highest priority issues (lowest scores)")
    print(f"   2. Run targeted fixes for failing components")
    print(f"   3. Re-run diagnostic to verify improvements")
    print(f"   4. Focus on getting overall score above 80%")

def main():
    """Run comprehensive diagnostic"""
    print("🚀 STARTING COMPREHENSIVE BU-PROCESSOR DIAGNOSTIC")
    print("=" * 80)
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Change to bu_processor directory if needed
    if Path("bu_processor").exists():
        os.chdir("bu_processor")
        print(f"📁 Changed to directory: {os.getcwd()}")
    
    # Run all diagnostic sections
    structure_ok = check_file_structure()
    import_results = check_python_environment()
    import_stats = run_core_imports()
    func_stats = run_basic_functionality()
    test_stats = run_test_suites()
    demo_stats = run_integration_demos()
    
    # Generate summary
    results = (structure_ok, import_results, import_stats, func_stats, test_stats, demo_stats)
    generate_summary(results)
    
    print(f"\n⏰ Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
