#!/usr/bin/env python3
"""Test Runner für BU-Processor mit verschiedenen Test-Modi."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


class TestRunner:
    """Zentraler Test Runner für alle Test-Kategorien."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tests_dir = project_root / "tests"
        
    def run_all_tests(self, verbose: bool = True, coverage: bool = True) -> int:
        """Führt alle Tests aus."""
        print("🧪 Führe alle Tests aus...")
        
        cmd = ["python", "-m", "pytest"]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend(["--cov=bu_processor", "--cov-report=term-missing"])
        
        cmd.append("tests/")
        
        return self._run_command(cmd)
    
    def run_unit_tests(self) -> int:
        """Führt nur Unit Tests aus."""
        print("🔧 Führe Unit Tests aus...")
        
        cmd = [
            "python", "-m", "pytest", 
            "-v",
            "-m", "unit",
            "tests/"
        ]
        
        return self._run_command(cmd)
    
    def run_integration_tests(self) -> int:
        """Führt nur Integration Tests aus."""
        print("🔗 Führe Integration Tests aus...")
        
        cmd = [
            "python", "-m", "pytest",
            "-v", 
            "-m", "integration",
            "tests/"
        ]
        
        return self._run_command(cmd)
    
    def run_specific_component(self, component: str) -> int:
        """Führt Tests für spezifische Komponente aus."""
        test_file = self.tests_dir / f"test_{component}.py"
        
        if not test_file.exists():
            print(f"❌ Test-Datei nicht gefunden: {test_file}")
            return 1
        
        print(f"🎯 Führe Tests für {component} aus...")
        
        cmd = [
            "python", "-m", "pytest",
            "-v",
            str(test_file)
        ]
        
        return self._run_command(cmd)
    
    def run_quick_tests(self) -> int:
        """Führt nur schnelle Tests aus (ohne langsame ML-Tests)."""
        print("⚡ Führe schnelle Tests aus...")
        
        cmd = [
            "python", "-m", "pytest",
            "-v",
            "-m", "not slow",
            "tests/"
        ]
        
        return self._run_command(cmd)
    
    def run_mock_tests_only(self) -> int:
        """Führt nur Tests mit Mocks aus (keine echten ML-Models)."""
        print("🎭 Führe Mock-Tests aus...")
        
        cmd = [
            "python", "-m", "pytest",
            "-v", 
            "-m", "mock",
            "tests/"
        ]
        
        return self._run_command(cmd)
    
    def run_with_coverage_report(self) -> int:
        """Führt Tests mit ausführlichem Coverage Report aus."""
        print("📊 Führe Tests mit Coverage Report aus...")
        
        cmd = [
            "python", "-m", "pytest",
            "-v",
            "--cov=bu_processor",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml", 
            "--cov-report=term-missing",
            "--cov-fail-under=70",  # Mindestens 70% Coverage
            "tests/"
        ]
        
        result = self._run_command(cmd)
        
        if result == 0:
            print("📈 Coverage Report erstellt:")
            print(f"   HTML: {self.project_root}/htmlcov/index.html")
            print(f"   XML:  {self.project_root}/coverage.xml")
        
        return result
    
    def run_performance_tests(self) -> int:
        """Führt Performance-Tests aus."""
        print("🏃 Führe Performance Tests aus...")
        
        cmd = [
            "python", "-m", "pytest",
            "-v",
            "--durations=20",  # Zeige 20 langsamste Tests
            "-k", "performance or batch or timing",
            "tests/"
        ]
        
        return self._run_command(cmd)
    
    def validate_test_setup(self) -> bool:
        """Validiert dass Test-Setup korrekt ist."""
        print("🔍 Validiere Test-Setup...")
        
        # Prüfe ob pytest installiert ist
        try:
            import pytest
            print(f"   ✅ pytest {pytest.__version__} gefunden")
        except ImportError:
            print("   ❌ pytest nicht installiert")
            return False
        
        # Prüfe ob pytest-mock installiert ist
        try:
            import pytest_mock
            print(f"   ✅ pytest-mock verfügbar")
        except ImportError:
            print("   ⚠️  pytest-mock nicht installiert (empfohlen)")
        
        # Prüfe ob pytest-cov installiert ist
        try:
            import pytest_cov
            print(f"   ✅ pytest-cov verfügbar")
        except ImportError:
            print("   ⚠️  pytest-cov nicht installiert (für Coverage)")
        
        # Prüfe Test-Verzeichnis
        if self.tests_dir.exists():
            test_files = list(self.tests_dir.glob("test_*.py"))
            print(f"   ✅ {len(test_files)} Test-Dateien gefunden")
            for test_file in test_files:
                print(f"      - {test_file.name}")
        else:
            print(f"   ❌ Tests-Verzeichnis nicht gefunden: {self.tests_dir}")
            return False
        
        # Prüfe ob BU-Processor package importiert werden kann
        try:
            sys.path.insert(0, str(self.project_root))
            import bu_processor
            print(f"   ✅ bu_processor package importierbar")
        except ImportError as e:
            print(f"   ❌ bu_processor package nicht importierbar: {e}")
            return False
        
        print("   ✅ Test-Setup validiert")
        return True
    
    def install_test_dependencies(self) -> int:
        """Installiert Test-Dependencies."""
        print("📦 Installiere Test-Dependencies...")
        
        cmd = [
            "pip", "install", "-r", "requirements-dev.txt"
        ]
        
        return self._run_command(cmd)
    
    def _run_command(self, cmd: List[str]) -> int:
        """Führt Kommando aus und gibt Exit-Code zurück."""
        try:
            print(f"   🔄 Ausführen: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.project_root, check=False)
            return result.returncode
        except Exception as e:
            print(f"   ❌ Kommando fehlgeschlagen: {e}")
            return 1


def main():
    """Hauptfunktion für Command Line Interface."""
    parser = argparse.ArgumentParser(description="BU-Processor Test Runner")
    
    parser.add_argument(
        "command",
        choices=[
            "all", "unit", "integration", "quick", "mock", 
            "coverage", "performance", "validate", "install-deps"
        ],
        help="Test-Kommando zum Ausführen"
    )
    
    parser.add_argument(
        "--component",
        choices=["classifier", "pdf_extractor", "pipeline_components"],
        help="Spezifische Komponente testen"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Coverage-Berichte deaktivieren"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true", 
        help="Weniger Output"
    )
    
    args = parser.parse_args()
    
    # Projekt-Root ermitteln
    current_dir = Path(__file__).parent
    project_root = current_dir.parent if current_dir.name == "scripts" else current_dir
    
    runner = TestRunner(project_root)
    
    # Command dispatch
    exit_code = 0
    
    if args.command == "validate":
        if not runner.validate_test_setup():
            exit_code = 1
    
    elif args.command == "install-deps":
        exit_code = runner.install_test_dependencies()
    
    elif args.component:
        exit_code = runner.run_specific_component(args.component)
    
    elif args.command == "all":
        exit_code = runner.run_all_tests(
            verbose=not args.quiet,
            coverage=not args.no_coverage
        )
    
    elif args.command == "unit":
        exit_code = runner.run_unit_tests()
    
    elif args.command == "integration":
        exit_code = runner.run_integration_tests()
    
    elif args.command == "quick":
        exit_code = runner.run_quick_tests()
    
    elif args.command == "mock":
        exit_code = runner.run_mock_tests_only()
    
    elif args.command == "coverage":
        exit_code = runner.run_with_coverage_report()
    
    elif args.command == "performance":
        exit_code = runner.run_performance_tests()
    
    # Ergebnis anzeigen
    if exit_code == 0:
        print("\n✅ Tests erfolgreich abgeschlossen!")
    else:
        print(f"\n❌ Tests fehlgeschlagen (Exit Code: {exit_code})")
    
    return exit_code


# Quick-Access Funktionen für direkte Nutzung
def run_all():
    """Direkte Funktion für alle Tests."""
    runner = TestRunner(Path(__file__).parent.parent)
    return runner.run_all_tests()


def run_component(component: str):
    """Direkte Funktion für Komponenten-Tests."""
    runner = TestRunner(Path(__file__).parent.parent)
    return runner.run_specific_component(component)


def validate_setup():
    """Direkte Funktion für Setup-Validierung."""
    runner = TestRunner(Path(__file__).parent.parent)
    return runner.validate_test_setup()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
