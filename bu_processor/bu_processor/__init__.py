"""
BU-Processor - Advanced ML Document Classification System
========================================================

A comprehensive ML-powered document classifier and processor specifically 
designed for insurance documents (Berufsunfähigkeitsversicherung).

Version: 0.1.0
"""

# bu_processor/__init__.py (MVP-clean)
__all__ = ["__version__"]
__version__ = "0.1.0"

# WICHTIG:
# KEINE Importe von Pipelines/Classifier/Config/Logging hier!
# Keine setup_logging()-Aufrufe, keine Konfiguration laden.
# Das passiert gezielt dort, wo es gebraucht wird (z. B. in API-Start).

# Später kannst du optional einen „lazy" Setup‑Hook anbieten (z. B. setup_logging()),
# aber niemals beim Import ausführen.
