#!/usr/bin/env python3
"""
Finale Projekt-Bereinigung für BU-Processor
==========================================
Entfernt alle redundanten Dateien systematisch.
"""

import os
import sys
from pathlib import Path
from typing import List
import glob

def print_status(message: str, success: bool = True):
    """Gibt eine farbige Statusmeldung aus."""
    emoji = "✅" if success else "❌"
    print(f"{emoji} {message}")

def safe_remove_file(file_path: Path) -> bool:
    """Entfernt eine Datei sicher und gibt bei Erfolg True zurück."""
    try:
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            print_status(f"Entfernt: {file_path.relative_to(file_path.parent.parent)}")
            return True
        else:
            # Kein Fehler, wenn die Datei schon weg ist
            return False
    except Exception as e:
        print_status(f"Fehler beim Entfernen von {file_path.name}: {e}", success=False)
        return False

def cleanup_project():
    """Hauptfunktion zur Bereinigung."""
    print("🧹 BU-Processor Finale Bereinigung")
    print("=" * 50)

    # 1. Projekt-Stammverzeichnis relativ zum Skript finden
    # Annahme: Das Skript liegt im `scripts`-Ordner
    project_root = Path(__file__).parent.parent
    if not (project_root / "pyproject.toml").exists():
        print_status(f"Projektverzeichnis nicht gefunden. Erwartet bei: {project_root}", success=False)
        return

    # Zähler für die Zusammenfassung
    counts = {"backup": 0, "docs": 0, "cli": 0}

    # 2. Backup-Dateien (.bak) automatisch finden und entfernen
    print("\n📦 Entferne Backup-Dateien (.bak):")
    # `rglob` durchsucht das Verzeichnis und alle Unterverzeichnisse
    backup_files = list(project_root.rglob("*.bak"))
    for file_path in backup_files:
        if safe_remove_file(file_path):
            counts["backup"] += 1
    print(f"   → {counts['backup']} Backup-Dateien entfernt")

    # 3. Veraltetes CLI-Skript entfernen
    print("\n🔧 Entferne veraltetes CLI-Skript:")
    old_cli = project_root / "cli.py"
    if old_cli.exists() and safe_remove_file(old_cli):
        counts["cli"] = 1
    print(f"   → {counts['cli']} veraltetes CLI-Skript entfernt (besseres CLI in bu_processor/cli.py)")

    # 4. Redundante Dokumentations-Dateien entfernen
    print("\n📄 Entferne redundante Dokumentation:")
    doc_files_to_remove = [
        "PDF_INTEGRATION_SUMMARY.md",
        "PINECONE_INTEGRATION_COMPLETE.md",
        "SEMANTIC_CHUNKING_INTEGRATION_COMPLETE.md",
        "CONTRIBUTORS.md"
    ]
    for file_name in doc_files_to_remove:
        file_path = project_root / file_name
        if file_path.exists() and safe_remove_file(file_path):
            counts["docs"] += 1
    print(f"   → {counts['docs']} redundante Dokumentationsdateien entfernt")

    print("\n✨ Bereinigung abgeschlossen!")
    print("📊 Zusammenfassung:")
    print(f"   • {counts['backup']} Backup-Dateien entfernt")
    print(f"   • {counts['docs']} redundante Dokumentationen entfernt")
    print(f"   • {counts['cli']} veraltetes CLI-Skript entfernt")

    print("\n🎯 Repository ist jetzt bereit für:")
    print("   • Git commit & push")
    print("   • Production deployment")
    print("   • Weitere Entwicklung")

if __name__ == "__main__":
    try:
        cleanup_project()
    except KeyboardInterrupt:
        print("\n⚠️  Bereinigung abgebrochen")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        sys.exit(1)