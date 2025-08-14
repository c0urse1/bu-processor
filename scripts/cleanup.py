#!/usr/bin/env python3
"""
Finale Projekt-Bereinigung für BU-Processor
==========================================
Entfernt alle redundanten Dateien systematisch.
Unterstützt --dry-run für sichere Vorschau.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List
import glob

def print_status(message: str, success: bool = True, dry_run: bool = False):
    """Gibt eine farbige Statusmeldung aus."""
    if dry_run:
        emoji = "🔍" if success else "⚠️"
        prefix = "[DRY RUN] "
    else:
        emoji = "✅" if success else "❌"
        prefix = ""
    print(f"{emoji} {prefix}{message}")

def safe_remove_file(file_path: Path, dry_run: bool = False) -> bool:
    """Entfernt eine Datei sicher und gibt bei Erfolg True zurück."""
    try:
        if file_path.exists() and file_path.is_file():
            if dry_run:
                print_status(f"Would remove: {file_path.relative_to(file_path.parent.parent)}", dry_run=True)
                return True
            else:
                file_path.unlink()
                print_status(f"Entfernt: {file_path.relative_to(file_path.parent.parent)}")
                return True
        else:
            # Kein Fehler, wenn die Datei schon weg ist
            return False
    except Exception as e:
        print_status(f"Fehler beim Entfernen von {file_path.name}: {e}", success=False, dry_run=dry_run)
        return False

def cleanup_project(dry_run: bool = False):
    """Hauptfunktion zur Bereinigung."""
    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"🧹 BU-Processor Finale Bereinigung ({mode})")
    print("=" * 50)

    # 1. Projekt-Stammverzeichnis relativ zum Skript finden
    # Annahme: Das Skript liegt im `scripts`-Ordner
    project_root = Path(__file__).parent.parent
    if not (project_root / "pyproject.toml").exists() and not (project_root / "bu_processor" / "pyproject.toml").exists():
        print_status(f"Projektverzeichnis nicht gefunden. Erwartet bei: {project_root}", success=False, dry_run=dry_run)
        return 1

    # Zähler für die Zusammenfassung
    counts = {"backup": 0, "docs": 0, "cli": 0, "test": 0}

    # 2. Backup-Dateien (.bak) automatisch finden und entfernen
    print("\n📦 Entferne Backup-Dateien (.bak):")
    backup_files = list(project_root.rglob("*.bak"))
    for file_path in backup_files:
        if safe_remove_file(file_path, dry_run):
            counts["backup"] += 1
    print(f"   → {counts['backup']} Backup-Dateien {'würden entfernt' if dry_run else 'entfernt'}")

    # 3. Veraltetes CLI-Skript entfernen
    print("\n🔧 Entferne veraltetes CLI-Skript:")
    old_cli = project_root / "cli.py"
    if old_cli.exists() and safe_remove_file(old_cli, dry_run):
        counts["cli"] = 1
    print(f"   → {counts['cli']} veraltetes CLI-Skript {'würde entfernt' if dry_run else 'entfernt'}")

    # 4. Redundante Test-Dateien im Root entfernen
    print("\n🧪 Entferne redundante Test-Dateien im Root:")
    test_patterns = [
        "test_*.py",
        "*_test.py", 
        "debug_*.py",
        "verify_*.py",
        "simple_*.py",
        "backup_*.py",
        "fix_*.py",
        "final_*.py",
        "minimal_*.py",
        "direct_*.py"
    ]
    
    for pattern in test_patterns:
        for file_path in project_root.glob(pattern):
            if file_path.is_file() and safe_remove_file(file_path, dry_run):
                counts["test"] += 1
    print(f"   → {counts['test']} redundante Test-Dateien {'würden entfernt' if dry_run else 'entfernt'}")

    # 5. Redundante Dokumentations-Dateien entfernen
    print("\n📄 Entferne redundante Dokumentation:")
    doc_files_to_remove = [
        "PDF_INTEGRATION_SUMMARY.md",
        "PINECONE_INTEGRATION_COMPLETE.md", 
        "SEMANTIC_CHUNKING_INTEGRATION_COMPLETE.md",
        "CONTRIBUTORS.md",
        "ENHANCED_SEMANTIC_API_COMPLETE.md",
        "SEMANTIC_ENHANCEMENT_FIXES_COMPLETE.md",
        "SIMHASH_GENERATOR_FIXES_COMPLETE.md",
        "TIMEOUT_RETRY_FIXES_COMPLETE.md"
    ]
    for file_name in doc_files_to_remove:
        file_path = project_root / file_name
        if file_path.exists() and safe_remove_file(file_path, dry_run):
            counts["docs"] += 1
    print(f"   → {counts['docs']} redundante Dokumentationsdateien {'würden entfernt' if dry_run else 'entfernt'}")

    print(f"\n✨ Bereinigung {'simuliert' if dry_run else 'abgeschlossen'}!")
    print("📊 Zusammenfassung:")
    verb = "würden entfernt" if dry_run else "entfernt"
    print(f"   • {counts['backup']} Backup-Dateien {verb}")
    print(f"   • {counts['test']} redundante Test-Dateien {verb}")
    print(f"   • {counts['docs']} redundante Dokumentationen {verb}")
    print(f"   • {counts['cli']} veraltetes CLI-Skript {verb}")

    if not dry_run:
        print("\n🎯 Repository ist jetzt bereit für:")
        print("   • Git commit & push")
        print("   • Production deployment")
    else:
        print(f"\n🎯 Führe ohne --dry-run aus, um {sum(counts.values())} Dateien zu entfernen")
        
    return 0

def main():
    """CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="BU-Processor Cleanup Script")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be removed without actually removing files")
    
    args = parser.parse_args()
    
    try:
        exit_code = cleanup_project(dry_run=args.dry_run)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n❌ Bereinigung abgebrochen durch Benutzer")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        sys.exit(1)
    print("   • Weitere Entwicklung")

if __name__ == "__main__":
    main()