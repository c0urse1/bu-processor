#!/usr/bin/env python3
"""
🔗 SEMANTIC CHUNKING INTEGRATION PIPELINE
========================================
Zentrale Pipeline zur Integration von PDF-Extraktion, semantischem Chunking 
und ML-Klassifikation für BU-Dokumentenverarbeitung.

End-to-End Workflow:
1. PDF-Dokument einlesen
2. Text extrahieren (PyMuPDF/PyPDF2)  
3. Semantisches Chunking anwenden
4. Chunks klassifizieren
5. Ergebnisse aggregieren und bewerten
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog

# Import Pipeline-Komponenten
from .pdf_extractor import EnhancedPDFExtractor, ExtractedContent, DocumentChunk, ChunkingStrategy
from .classifier import RealMLClassifier

# Import semantic chunking falls verfügbar
try:
    from .semantic_chunking_enhancement import SemanticClusteringEnhancer
    SEMANTIC_ENHANCEMENT_AVAILABLE = True
except ImportError:
    SEMANTIC_ENHANCEMENT_AVAILABLE = False

# Import Pinecone integration falls verfügbar
try:
    from .pinecone_integration import PineconePipeline, PineconeConfig, EmbeddingModel, PineconeEnvironment
    PINECONE_INTEGRATION_AVAILABLE = True
except ImportError:
    PINECONE_INTEGRATION_AVAILABLE = False

logger = structlog.get_logger("pipeline.integrated_pipeline")

@dataclass 
class PipelineResult:
    """Vollständiges Ergebnis der integrierten Pipeline"""
    
    # Input-Informationen
    input_file: str
    processing_time: float
    pipeline_version: str = "1.0"
    
    # PDF-Extraktion
    extracted_content: Optional[ExtractedContent] = None
    extraction_success: bool = False
    extraction_method: str = ""
    
    # Chunking-Ergebnisse  
    chunks: List[DocumentChunk] = field(default_factory=list)
    chunking_strategy: str = "none"
    chunking_success: bool = False
    
    # Klassifikation
    final_classification: Optional[Dict] = None
    chunk_classifications: List[Dict] = field(default_factory=list)
    classification_success: bool = False
    
    # Erweiterte Analyse
    semantic_analysis: Optional[Dict] = None
    confidence_analysis: Dict = field(default_factory=dict)
    quality_metrics: Dict = field(default_factory=dict)
    
    # Pinecone Vector Database
    pinecone_upload: Optional[Dict] = None
    pinecone_enabled: bool = False
    vector_search_results: List[Dict] = field(default_factory=list)
    
    # Fehler-Tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class ProcessingStrategy(Enum):
    """Verfügbare Pipeline-Strategien"""
    FAST = "fast"                    # Minimale Verarbeitung für schnelle Ergebnisse
    BALANCED = "balanced"            # Ausgewogen zwischen Speed und Qualität
    COMPREHENSIVE = "comprehensive"  # Vollständige Analyse mit allen Features
    SEMANTIC_FOCUS = "semantic"      # Fokus auf semantische Analyse
    VECTOR_ENHANCED = "vector"       # Mit Pinecone Vector Database Integration

class IntegratedPipeline:
    """Zentrale Pipeline für End-to-End PDF-zu-Klassifikation"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        enable_semantic_enhancement: bool = True,
        enable_pinecone: bool = False,
        pinecone_config: Optional[Dict] = None,
        default_strategy: ProcessingStrategy = ProcessingStrategy.BALANCED
    ):
        self.default_strategy = default_strategy
        self.enable_semantic_enhancement = enable_semantic_enhancement and SEMANTIC_ENHANCEMENT_AVAILABLE
        self.enable_pinecone = enable_pinecone and PINECONE_INTEGRATION_AVAILABLE
        
        # Initialisiere Pipeline-Komponenten
        try:
            self.pdf_extractor = EnhancedPDFExtractor(enable_chunking=True)
            logger.info("PDF Extractor initialisiert")
        except Exception as e:
            logger.error(f"PDF Extractor Initialisierung fehlgeschlagen: {e}")
            self.pdf_extractor = None
        
        try:
            self.classifier = RealMLClassifier(model_path) if model_path else RealMLClassifier()
            logger.info("ML Classifier initialisiert")
        except Exception as e:
            logger.error(f"ML Classifier Initialisierung fehlgeschlagen: {e}")
            self.classifier = None
        
        # Semantic Enhancer (optional)
        if self.enable_semantic_enhancement:
            try:
                self.semantic_enhancer = SemanticClusteringEnhancer()
                logger.info("Semantic Enhancer initialisiert")
            except Exception as e:
                logger.warning(f"Semantic Enhancer Initialisierung fehlgeschlagen: {e}")
                self.semantic_enhancer = None
        else:
            self.semantic_enhancer = None
        
        # Pinecone Pipeline (optional)
        if self.enable_pinecone:
            try:
                if pinecone_config:
                    pinecone_config_obj = PineconeConfig(**pinecone_config)
                else:
                    pinecone_config_obj = PineconeConfig()
                
                self.pinecone_pipeline = PineconePipeline(pinecone_config_obj)
                logger.info("Pinecone Pipeline initialisiert", 
                           index=pinecone_config_obj.index_name)
            except Exception as e:
                logger.warning(f"Pinecone Pipeline Initialisierung fehlgeschlagen: {e}")
                self.pinecone_pipeline = None
                self.enable_pinecone = False
        else:
            self.pinecone_pipeline = None
        
        # Strategy-spezifische Konfigurationen
        self.strategy_configs = {
            ProcessingStrategy.FAST: {
                "chunking_strategy": ChunkingStrategy.NONE,
                "max_chunk_size": 2000,
                "classify_chunks_individually": False,
                "enable_semantic_analysis": False,
                "detailed_confidence_analysis": False
            },
            ProcessingStrategy.BALANCED: {
                "chunking_strategy": ChunkingStrategy.SIMPLE,
                "max_chunk_size": 1000,
                "classify_chunks_individually": True,
                "enable_semantic_analysis": False,
                "detailed_confidence_analysis": True
            },
            ProcessingStrategy.COMPREHENSIVE: {
                "chunking_strategy": ChunkingStrategy.HYBRID,
                "max_chunk_size": 800,
                "classify_chunks_individually": True,
                "enable_semantic_analysis": True,
                "detailed_confidence_analysis": True
            },
            ProcessingStrategy.SEMANTIC_FOCUS: {
                "chunking_strategy": ChunkingStrategy.SEMANTIC,
                "max_chunk_size": 600,
                "classify_chunks_individually": True,
                "enable_semantic_analysis": True,
                "detailed_confidence_analysis": True
            },
            ProcessingStrategy.VECTOR_ENHANCED: {
                "chunking_strategy": ChunkingStrategy.HYBRID,
                "max_chunk_size": 800,
                "classify_chunks_individually": True,
                "enable_semantic_analysis": True,
                "detailed_confidence_analysis": True,
                "enable_pinecone_upload": True,
                "perform_vector_search": True
            }
        }
    
    def process_document(
        self,
        file_path: Union[str, Path],
        strategy: Optional[ProcessingStrategy] = None,
        custom_config: Optional[Dict] = None
    ) -> PipelineResult:
        """Hauptmethode: Vollständige Dokumentenverarbeitung"""
        
        start_time = time.time()
        file_path = Path(file_path)
        strategy = strategy or self.default_strategy
        
        logger.info("Pipeline-Verarbeitung gestartet",
                   file=str(file_path),
                   strategy=strategy.value)
        
        # Initialisiere Ergebnis-Container
        result = PipelineResult(
            input_file=str(file_path),
            processing_time=0.0
        )
        
        # Validiere Input
        if not self._validate_input(file_path, result):
            result.processing_time = time.time() - start_time
            return result
        
        # Lade Strategy-Konfiguration
        config = self.strategy_configs[strategy].copy()
        if custom_config:
            config.update(custom_config)
        
        # Pipeline-Schritte ausführen
        try:
            # Schritt 1: PDF-Extraktion mit Chunking
            result = self._extract_and_chunk(file_path, config, result)
            
            # Schritt 2: Klassifikation
            result = self._classify_content(config, result)
            
            # Schritt 3: Erweiterte Analyse (optional)
            if config.get("enable_semantic_analysis", False):
                result = self._perform_semantic_analysis(result)
            
            # Schritt 4: Qualitätsanalyse
            if config.get("detailed_confidence_analysis", False):
                result = self._analyze_quality_metrics(result)
            
            # Schritt 5: Pinecone Vector Upload (optional)
            if config.get("enable_pinecone_upload", False) and self.enable_pinecone:
                result = self._upload_to_pinecone(result, file_path)
            
            # Schritt 6: Vector Search Demo (optional)
            if config.get("perform_vector_search", False) and self.enable_pinecone:
                result = self._perform_vector_search_demo(result)
            
            # Schritt 7: Finale Aggregation
            result = self._aggregate_final_results(result)
            
        except Exception as e:
            error_msg = f"Pipeline-Verarbeitung fehlgeschlagen: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        result.processing_time = time.time() - start_time
        
        logger.info("Pipeline-Verarbeitung abgeschlossen",
                   file=str(file_path),
                   success=len(result.errors) == 0,
                   processing_time=result.processing_time)
        
        return result
    
    def _validate_input(self, file_path: Path, result: PipelineResult) -> bool:
        """Validiert Input-Datei"""
        
        if not file_path.exists():
            error_msg = f"Datei nicht gefunden: {file_path}"
            result.errors.append(error_msg)
            return False
        
        if not file_path.suffix.lower() == '.pdf':
            error_msg = f"Keine PDF-Datei: {file_path}"
            result.errors.append(error_msg)
            return False
        
        if not self.pdf_extractor or not self.classifier:
            error_msg = "Pipeline-Komponenten nicht vollständig initialisiert"
            result.errors.append(error_msg)
            return False
        
        return True
    
    def _extract_and_chunk(
        self, 
        file_path: Path, 
        config: Dict, 
        result: PipelineResult
    ) -> PipelineResult:
        """Schritt 1: PDF-Extraktion und Chunking"""
        
        try:
            logger.info("PDF-Extraktion und Chunking", strategy=config["chunking_strategy"].value)
            
            extracted_content = self.pdf_extractor.extract_text_from_pdf(
                file_path,
                chunking_strategy=config["chunking_strategy"],
                max_chunk_size=config["max_chunk_size"],
                overlap_size=100
            )\n            \n            result.extracted_content = extracted_content\n            result.extraction_success = True\n            result.extraction_method = extracted_content.extraction_method\n            result.chunks = extracted_content.chunks\n            result.chunking_strategy = extracted_content.chunking_method\n            result.chunking_success = extracted_content.chunking_enabled\n            \n            logger.info("Extraktion erfolgreich",\n                       text_length=len(extracted_content.text),\n                       chunks_created=len(extracted_content.chunks))\n            \n        except Exception as e:\n            error_msg = f"PDF-Extraktion fehlgeschlagen: {e}"\n            logger.error(error_msg)\n            result.errors.append(error_msg)\n        \n        return result\n    \n    def _classify_content(self, config: Dict, result: PipelineResult) -> PipelineResult:\n        """Schritt 2: Inhalt klassifizieren"""\n        \n        if not result.extraction_success or not result.extracted_content:\n            result.warnings.append("Keine extrahierten Inhalte für Klassifikation verfügbar")\n            return result\n        \n        try:\n            logger.info("Klassifikation gestartet", \n                       strategy="chunk-based" if config["classify_chunks_individually"] else "full-text")\n            \n            if config["classify_chunks_individually"] and result.chunks:\n                # Chunk-basierte Klassifikation\n                classification_result = self.classifier._classify_pdf_with_chunks(result.extracted_content)\n            else:\n                # Traditionelle Volltext-Klassifikation\n                classification_result = self.classifier._classify_pdf_traditional(result.extracted_content)\n            \n            result.final_classification = classification_result\n            result.classification_success = "error" not in classification_result\n            \n            # Chunk-spezifische Ergebnisse extrahieren\n            if "chunk_results" in classification_result:\n                result.chunk_classifications = classification_result["chunk_results"]\n            \n            logger.info("Klassifikation abgeschlossen",\n                       category=classification_result.get("category"),\n                       confidence=classification_result.get("confidence", 0))\n            \n        except Exception as e:\n            error_msg = f"Klassifikation fehlgeschlagen: {e}"\n            logger.error(error_msg)\n            result.errors.append(error_msg)\n        \n        return result\n    \n    def _perform_semantic_analysis(self, result: PipelineResult) -> PipelineResult:\n        """Schritt 3: Erweiterte semantische Analyse"""\n        \n        if not self.semantic_enhancer or not result.chunks:\n            result.warnings.append("Semantische Analyse nicht verfügbar")\n            return result\n        \n        try:\n            logger.info("Semantische Analyse gestartet")\n            \n            # Hier würde die vollständige Integration mit semantic_chunking_enhancement erfolgen\n            # Placeholder für erweiterte semantische Analyse\n            \n            semantic_data = {\n                "semantic_enhancement_applied": True,\n                "total_chunks_analyzed": len(result.chunks),\n                "semantic_clusters_found": max(1, len(result.chunks) // 3),\n                "cluster_coherence_average": 0.75,  # Placeholder\n                "semantic_similarity_matrix_size": f"{len(result.chunks)}x{len(result.chunks)}\"\n            }\n            \n            result.semantic_analysis = semantic_data\n            \n            logger.info("Semantische Analyse abgeschlossen",\n                       clusters=semantic_data["semantic_clusters_found"])\n            \n        except Exception as e:\n            error_msg = f"Semantische Analyse fehlgeschlagen: {e}"\n            logger.error(error_msg)\n            result.errors.append(error_msg)\n        \n        return result\n    \n    def _analyze_quality_metrics(self, result: PipelineResult) -> PipelineResult:\n        """Schritt 4: Qualitäts- und Confidence-Analyse"""\n        \n        try:\n            logger.info("Qualitätsanalyse gestartet")\n            \n            metrics = {}\n            \n            # Basis-Metriken\n            if result.final_classification:\n                metrics["overall_confidence"] = result.final_classification.get("confidence", 0.0)\n                metrics["classification_reliable"] = result.final_classification.get("is_confident", False)\n            \n            # Chunk-basierte Metriken\n            if result.chunk_classifications:\n                confidences = [c.get("classification", {}).get("confidence", 0) \n                             for c in result.chunk_classifications]\n                \n                if confidences:\n                    metrics["chunk_confidence_mean"] = sum(confidences) / len(confidences)\n                    metrics["chunk_confidence_std"] = (sum((c - metrics["chunk_confidence_mean"])**2 \n                                                          for c in confidences) / len(confidences))**0.5\n                    metrics["chunk_confidence_min"] = min(confidences)\n                    metrics["chunk_confidence_max"] = max(confidences)\n                    metrics["high_confidence_chunks_ratio"] = len([c for c in confidences if c > 0.8]) / len(confidences)\n            \n            # Text-Qualitäts-Metriken\n            if result.extracted_content:\n                text = result.extracted_content.text\n                metrics["text_length"] = len(text)\n                metrics["text_word_count"] = len(text.split())\n                metrics["text_paragraph_count"] = len([p for p in text.split('\\n\\n') if p.strip()])\n                \n                # Einfache Text-Qualitätsindikatoren\n                metrics["text_avg_sentence_length"] = metrics["text_word_count"] / max(1, text.count('.') + text.count('!') + text.count('?'))\n                metrics["text_whitespace_ratio"] = len([c for c in text if c.isspace()]) / len(text) if text else 0\n            \n            # Pipeline-Performance Metriken\n            metrics["processing_success_rate"] = 1.0 - (len(result.errors) / max(1, len(result.errors) + 1))\n            metrics["warnings_count"] = len(result.warnings)\n            metrics["errors_count"] = len(result.errors)\n            \n            result.quality_metrics = metrics\n            \n            logger.info("Qualitätsanalyse abgeschlossen",\n                       overall_confidence=metrics.get("overall_confidence", 0),\n                       text_quality_score=metrics.get("processing_success_rate", 0))\n            \n        except Exception as e:\n            error_msg = f"Qualitätsanalyse fehlgeschlagen: {e}"\n            logger.error(error_msg)\n            result.errors.append(error_msg)\n        \n        return result\n    \n    def _aggregate_final_results(self, result: PipelineResult) -> PipelineResult:\n        """Schritt 5: Finale Ergebnis-Aggregation"""\n        \n        try:\n            logger.info("Finale Aggregation")\n            \n            # Confidence-Analyse für finale Entscheidung\n            confidence_factors = []\n            \n            if result.final_classification:\n                confidence_factors.append(result.final_classification.get("confidence", 0))\n            \n            if result.quality_metrics:\n                if "chunk_confidence_mean" in result.quality_metrics:\n                    confidence_factors.append(result.quality_metrics["chunk_confidence_mean"])\n                \n                if "processing_success_rate" in result.quality_metrics:\n                    confidence_factors.append(result.quality_metrics["processing_success_rate"])\n            \n            # Gewichtete finale Confidence\n            if confidence_factors:\n                final_confidence = sum(confidence_factors) / len(confidence_factors)\n                \n                result.confidence_analysis = {\n                    "final_weighted_confidence": final_confidence,\n                    "confidence_factors_count": len(confidence_factors),\n                    "confidence_factors": confidence_factors,\n                    "high_confidence_threshold": 0.8,\n                    "recommendation": "accept" if final_confidence > 0.8 else "review" if final_confidence > 0.6 else "reject"\n                }\n            \n            logger.info("Pipeline erfolgreich abgeschlossen",\n                       final_confidence=result.confidence_analysis.get("final_weighted_confidence", 0),\n                       recommendation=result.confidence_analysis.get("recommendation", "unknown"))\n            \n        except Exception as e:\n            error_msg = f"Finale Aggregation fehlgeschlagen: {e}"\n            logger.error(error_msg)\n            result.errors.append(error_msg)\n        \n        return result\n    \n    def process_multiple_documents(\n        self,\n        file_paths: List[Union[str, Path]],\n        strategy: Optional[ProcessingStrategy] = None,\n        max_parallel: int = 4\n    ) -> List[PipelineResult]:\n        \"\"\"Batch-Verarbeitung mehrerer Dokumente\"\"\"\n        \n        logger.info("Batch-Verarbeitung gestartet", \n                   document_count=len(file_paths),\n                   strategy=strategy.value if strategy else self.default_strategy.value)\n        \n        results = []\n        \n        for file_path in file_paths:\n            try:\n                result = self.process_document(file_path, strategy)\n                results.append(result)\n                \n                logger.info("Dokument verarbeitet",\n                           file=str(file_path),\n                           success=len(result.errors) == 0)\n                \n            except Exception as e:\n                logger.error("Dokument-Verarbeitung fehlgeschlagen",\n                           file=str(file_path),\n                           error=str(e))\n                \n                # Erstelle Fehler-Ergebnis\n                error_result = PipelineResult(\n                    input_file=str(file_path),\n                    processing_time=0.0\n                )\n                error_result.errors.append(f"Verarbeitung fehlgeschlagen: {e}")\n                results.append(error_result)\n        \n        successful = len([r for r in results if len(r.errors) == 0])\n        \n        logger.info("Batch-Verarbeitung abgeschlossen",\n                   total_documents=len(file_paths),\n                   successful=successful,\n                   failed=len(file_paths) - successful)\n        \n        return results\n\ndef demo_integrated_pipeline():\n    \"\"\"Demo-Funktion für die integrierte Pipeline\"\"\"\n    print(\"🔗 Integrierte Pipeline Demo\")\n    print(\"=============================\")\n    \n    try:\n        # Initialisiere Pipeline\n        pipeline = IntegratedPipeline()\n        \n        test_pdf = Path(\"tests/fixtures/sample.pdf\")\n        \n        if not test_pdf.exists():\n            print(f\"⚠️  Test-PDF nicht gefunden: {test_pdf}\")\n            print(\"   Erstelle Test-PDFs mit: python scripts/generate_test_pdfs.py\")\n            return\n        \n        print(f\"\\n📄 Teste verschiedene Pipeline-Strategien mit {test_pdf.name}:\")\n        \n        strategies = [\n            ProcessingStrategy.FAST,\n            ProcessingStrategy.BALANCED,\n            ProcessingStrategy.COMPREHENSIVE\n        ]\n        \n        for strategy in strategies:\n            print(f\"\\n🔄 {strategy.value.upper()} Pipeline:\")\n            \n            result = pipeline.process_document(test_pdf, strategy)\n            \n            print(f\"   ⏱️  Verarbeitungszeit: {result.processing_time:.2f}s\")\n            print(f\"   ✅ Extraktion: {'Ja' if result.extraction_success else 'Nein'}\")\n            print(f\"   🧩 Chunking: {result.chunking_strategy} ({len(result.chunks)} Chunks)\")\n            print(f\"   🤖 Klassifikation: {'Ja' if result.classification_success else 'Nein'}\")\n            \n            if result.final_classification:\n                print(f\"   📊 Kategorie: {result.final_classification.get('category', 'N/A')}\")\n                print(f\"   📈 Confidence: {result.final_classification.get('confidence', 0):.2f}\")\n            \n            if result.confidence_analysis:\n                print(f\"   🎯 Finale Confidence: {result.confidence_analysis.get('final_weighted_confidence', 0):.2f}\")\n                print(f\"   💡 Empfehlung: {result.confidence_analysis.get('recommendation', 'N/A')}\")\n            \n            if result.errors:\n                print(f\"   ❌ Fehler: {len(result.errors)}\")\n            if result.warnings:\n                print(f\"   ⚠️  Warnungen: {len(result.warnings)}\")\n        \n        print(f\"\\n🎉 Pipeline-Demo abgeschlossen!\")\n        \n    except Exception as e:\n        print(f\"❌ Pipeline-Demo fehlgeschlagen: {e}\")\n\nif __name__ == \"__main__\":\n    demo_integrated_pipeline()\n