#!/usr/bin/env python3
"""
Code Quality Runner Script
==========================

Führt alle Code-Qualitäts-Tools in der richtigen Reihenfolge aus:
1. isort - Import-Sortierung
2. black - Code-Formatierung  
3. flake8 - Linting
4. mypy - Type-Checking

Usage:
    python scripts/code_quality.py [--check] [--fix] [--mypy-only]
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


class CodeQualityRunner:
    """Runner für Code-Qualitäts-Tools."""
    
    def __init__(self, project_root: Path) -> None:
        """Initialisiert den Code Quality Runner.
        
        Args:
            project_root: Pfad zum Projekt-Root-Verzeichnis
        """
        self.project_root = project_root
        self.src_dirs = ["bu_processor", "tests", "scripts"]
        
    def run_command(self, command: List[str], description: str) -> bool:
        """Führt ein Kommandozeilen-Kommando aus.
        
        Args:
            command: Liste der Kommando-Argumente
            description: Beschreibung für Logging
            
        Returns:
            True bei erfolgreichem Exit-Code
        """
        print(f"🔧 {description}...")
        print(f"   Command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 Minuten timeout
            )
            
            if result.returncode == 0:
                print(f"   ✅ {description} - SUCCESS")
                if result.stdout.strip():
                    print(f"   Output: {result.stdout.strip()}")
                return True
            else:
                print(f"   ❌ {description} - FAILED (Exit code: {result.returncode})")
                if result.stdout.strip():
                    print(f"   Stdout: {result.stdout.strip()}")
                if result.stderr.strip():
                    print(f"   Stderr: {result.stderr.strip()}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ {description} - TIMEOUT")
            return False
        except FileNotFoundError:
            print(f"   ❌ {description} - TOOL NOT FOUND")
            return False
        except Exception as e:
            print(f"   ❌ {description} - ERROR: {e}")
            return False
    
    def run_isort(self, check_only: bool = False) -> bool:
        """Führt isort aus.
        
        Args:
            check_only: Nur prüfen ohne Änderungen
            
        Returns:
            True bei Erfolg
        """
        command = ["python", "-m", "isort"]
        
        if check_only:
            command.extend(["--check", "--diff"])
            
        command.extend(self.src_dirs)
        
        return self.run_command(command, "Import sorting with isort")
    
    def run_black(self, check_only: bool = False) -> bool:
        """Führt black aus.
        
        Args:
            check_only: Nur prüfen ohne Änderungen
            
        Returns:
            True bei Erfolg
        """
        command = ["python", "-m", "black"]
        
        if check_only:
            command.extend(["--check", "--diff"])
            
        command.extend(self.src_dirs)
        
        return self.run_command(command, "Code formatting with black")
    
    def run_flake8(self) -> bool:
        """Führt flake8 aus.
        
        Returns:
            True bei Erfolg
        """
        command = ["python", "-m", "flake8"] + self.src_dirs
        
        return self.run_command(command, "Linting with flake8")
    
    def run_mypy(self) -> bool:
        """Führt mypy aus.
        
        Returns:
            True bei Erfolg
        """
        # MyPy auf Hauptmodule anwenden
        command = ["python", "-m", "mypy", "bu_processor"]
        
        return self.run_command(command, "Type checking with mypy")
    
    def install_dev_dependencies(self) -> bool:
        """Installiert Development-Dependencies.
        
        Returns:
            True bei Erfolg
        """
        command = ["pip", "install", "-r", "requirements-dev.txt"]
        
        return self.run_command(command, "Installing dev dependencies")
    
    def run_all(
        self, 
        check_only: bool = False, 
        fix_mode: bool = False,
        mypy_only: bool = False,
        install_deps: bool = False
    ) -> bool:
        """Führt alle Tools aus.
        
        Args:
            check_only: Nur prüfen ohne Änderungen
            fix_mode: Automatisch reparieren wo möglich
            mypy_only: Nur MyPy ausführen
            install_deps: Zuerst Dependencies installieren
            
        Returns:
            True wenn alle Tools erfolgreich waren
        """
        print("🚀 Code Quality Check gestartet")
        print(f"   Project Root: {self.project_root}")
        print(f"   Source Dirs: {', '.join(self.src_dirs)}")
        print(f"   Check Only: {check_only}")
        print(f"   Fix Mode: {fix_mode}")
        print(f"   MyPy Only: {mypy_only}")
        print()
        
        success = True
        
        # Dependencies installieren falls gewünscht
        if install_deps:
            if not self.install_dev_dependencies():
                print("⚠️  Dependency installation failed, continuing anyway...")
        
        if mypy_only:
            # Nur MyPy ausführen
            success = self.run_mypy()
        else:
            # Vollständiger Workflow
            
            # 1. isort - Import-Sortierung
            if not self.run_isort(check_only=check_only):
                success = False
                if not fix_mode:
                    print("💡 Tipp: Führe 'isort bu_processor tests scripts' aus um zu reparieren")
            
            # 2. black - Code-Formatierung
            if not self.run_black(check_only=check_only):
                success = False
                if not fix_mode:
                    print("💡 Tipp: Führe 'black bu_processor tests scripts' aus um zu reparieren")
            
            # 3. flake8 - Linting
            if not self.run_flake8():
                success = False
                print("💡 Tipp: Flake8-Fehler müssen manuell behoben werden")
            
            # 4. mypy - Type Checking
            if not self.run_mypy():
                success = False
                print("💡 Tipp: MyPy-Fehler erfordern Type Hints oder Konfigurationsanpassungen")
        
        print()
        if success:
            print("🎉 Alle Code-Qualitäts-Checks erfolgreich!")
            return True
        else:
            print("❌ Ein oder mehrere Code-Qualitäts-Checks fehlgeschlagen")
            return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="BU-Processor Code Quality Runner")
    
    parser.add_argument(
        "--check", 
        action="store_true", 
        help="Nur prüfen, keine Änderungen vornehmen"
    )
    
    parser.add_argument(
        "--fix", 
        action="store_true", 
        help="Automatisch reparieren wo möglich"
    )
    
    parser.add_argument(
        "--mypy-only", 
        action="store_true", 
        help="Nur MyPy Type-Checking ausführen"
    )
    
    parser.add_argument(
        "--install-deps", 
        action="store_true", 
        help="Zuerst Development-Dependencies installieren"
    )
    
    args = parser.parse_args()
    
    # Projekt-Root finden
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Validierung
    if not (project_root / "bu_processor").exists():
        print(f"❌ bu_processor Modul nicht gefunden in {project_root}")
        return 1
    
    # Runner erstellen und ausführen
    runner = CodeQualityRunner(project_root)
    
    success = runner.run_all(
        check_only=args.check,
        fix_mode=args.fix, 
        mypy_only=args.mypy_only,
        install_deps=args.install_deps
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
