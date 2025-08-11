print("--- Starte einfachen Import-Test ---")
try:
    from bu_processor.pipeline.classifier import RealMLClassifier
    print("✅ ERFOLG: RealMLClassifier konnte erfolgreich importiert werden!")
except ModuleNotFoundError as e:
    print(f"❌ FEHLER: Konnte immer noch nicht importieren. Fehler: {e}")
except Exception as e:
    print(f"❌ FEHLER mit einem anderen Problem: {e}")
print("--- Test beendet ---")