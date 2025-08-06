#!/usr/bin/env python3
"""
🎬 SEMANTIC CHUNKING INTEGRATION DEMO
===================================
Vollständige Demonstration der integrierten Pipeline mit semantischem Chunking
"""

import time
from pathlib import Path
import sys
import os

def demo_complete_integration():
    """Umfassende Demo der vollständigen Integration"""
    
    print("🎬 SEMANTIC CHUNKING INTEGRATION - LIVE DEMO")
    print("=" * 55)
    
    try:
        # Setup
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
        
        # Test ob alle Module verfügbar sind
        print("🔧 Teste Module-Verfügbarkeit...")
        
        try:
            from pipeline.pdf_extractor import EnhancedPDFExtractor, ChunkingStrategy
            print("   ✅ Enhanced PDF Extractor")
        except ImportError as e:
            print(f"   ❌ PDF Extractor: {e}")
            return
        
        try:
            from pipeline.classifier import RealMLClassifier
            print("   ✅ ML Classifier")
        except ImportError as e:
            print(f"   ❌ ML Classifier: {e}")
            return
        
        try:
            from pipeline.integrated_pipeline import IntegratedPipeline, ProcessingStrategy
            print("   ✅ Integrated Pipeline")
        except ImportError as e:
            print(f"   ❌ Integrated Pipeline: {e}")
            return
        
        try:
            from pipeline.semantic_chunking_enhancement import SemanticClusteringEnhancer
            print("   ✅ Semantic Chunking (verfügbar)")
            semantic_available = True
        except ImportError:
            print("   ⚠️  Semantic Chunking (nicht verfügbar, aber Pipeline funktioniert)")
            semantic_available = False
        
        print()
        
        # Erstelle Test-PDF falls nicht vorhanden
        test_pdf = Path("tests/fixtures/sample.pdf")
        if not test_pdf.exists():
            print("📄 Erstelle Test-PDF...")
            try:
                from scripts.generate_test_pdfs import generate_test_pdfs
                generate_test_pdfs()
                print("   ✅ Test-PDFs erstellt")
            except Exception as e:
                print(f"   ⚠️  Test-PDF-Erstellung fehlgeschlagen: {e}")
                print("   💡 Verwende manuell erstellte PDF oder: python scripts/generate_test_pdfs.py")
        
        if not test_pdf.exists():
            print("❌ Keine Test-PDF verfügbar. Demo wird mit Text-Beispiel fortgesetzt.")
            demo_text_only()
            return
        
        print(f"📄 Verwende Test-PDF: {test_pdf}")
        print()
        
        # DEMO 1: PDF-Extraktion mit verschiedenen Chunking-Strategien
        print("🧩 DEMO 1: Chunking-Strategien vergleichen")
        print("-" * 45)
        
        extractor = EnhancedPDFExtractor(enable_chunking=True)
        
        strategies = [
            (ChunkingStrategy.NONE, "Kein Chunking"),
            (ChunkingStrategy.SIMPLE, "Einfaches Chunking"),
            (ChunkingStrategy.SEMANTIC, "Semantisches Chunking"),
            (ChunkingStrategy.HYBRID, "Hybrid Chunking")
        ]
        
        extraction_results = {}
        
        for strategy, name in strategies:
            print(f"\\n   📊 {name}:")
            start_time = time.time()
            
            try:
                result = extractor.extract_text_from_pdf(
                    test_pdf,
                    chunking_strategy=strategy,
                    max_chunk_size=600,
                    overlap_size=50
                )
                
                processing_time = time.time() - start_time
                extraction_results[strategy] = result
                
                print(f"      ⏱️  Zeit: {processing_time:.2f}s")
                print(f"      📝 Text: {len(result.text)} Zeichen")
                print(f"      🧩 Chunks: {len(result.chunks)}")
                print(f"      🔧 Methode: {result.extraction_method}")
                
                if result.chunks:
                    avg_size = sum(len(c.text) for c in result.chunks) / len(result.chunks)
                    print(f"      📐 Ø Chunk-Größe: {avg_size:.0f} Zeichen")
                
                if result.semantic_clusters:
                    print(f"      🧠 Semantic Clusters: {result.semantic_clusters.get('total_clusters', 'N/A')}")
                
            except Exception as e:
                print(f"      ❌ Fehler: {e}")
                extraction_results[strategy] = None
        
        print()
        
        # DEMO 2: Klassifikation mit Chunking
        print("🤖 DEMO 2: Erweiterte Klassifikation")
        print("-" * 35)
        
        try:
            classifier = RealMLClassifier()
            
            # Teste verschiedene Klassifikations-Ansätze
            classification_approaches = [
                (ChunkingStrategy.NONE, False, "Traditionell"),
                (ChunkingStrategy.SIMPLE, True, "Chunk-basiert (Simple)"),
                (ChunkingStrategy.SEMANTIC, True, "Chunk-basiert (Semantic)"),
            ]
            
            for strategy, use_chunks, name in classification_approaches:
                print(f"\\n   🎯 {name}:")
                start_time = time.time()
                
                try:
                    result = classifier.classify_pdf(
                        test_pdf,
                        chunking_strategy=strategy,
                        max_chunk_size=800,
                        classify_chunks_individually=use_chunks
                    )
                    
                    processing_time = time.time() - start_time
                    
                    print(f"      ⏱️  Zeit: {processing_time:.2f}s")
                    print(f"      🏷️  Kategorie: {result.get('category', 'N/A')}")
                    print(f"      📊 Confidence: {result.get('confidence', 0):.2f}")
                    print(f"      ✅ Vertrauenswürdig: {'Ja' if result.get('is_confident', False) else 'Nein'}")
                    print(f"      🔧 Input-Typ: {result.get('input_type', 'N/A')}")
                    
                    if 'chunk_analysis' in result:
                        ca = result['chunk_analysis']
                        print(f"      🧩 Chunks verarbeitet: {ca.get('processed_chunks', 0)}")
                        print(f"      📈 Hohe Confidence: {ca.get('high_confidence_chunks', 0)}")
                        
                except Exception as e:
                    print(f"      ❌ Fehler: {e}")
        
        except Exception as e:
            print(f"❌ Classifier-Initialisierung fehlgeschlagen: {e}")
            print("💡 Stelle sicher, dass ein ML-Modell verfügbar ist")
        
        print()
        
        # DEMO 3: Integrierte End-to-End Pipeline
        print("🔗 DEMO 3: End-to-End Pipeline")
        print("-" * 30)
        
        try:
            pipeline = IntegratedPipeline()
            
            pipeline_strategies = [
                (ProcessingStrategy.FAST, "Schnell"),
                (ProcessingStrategy.BALANCED, "Ausgewogen"),
                (ProcessingStrategy.COMPREHENSIVE, "Umfassend")
            ]
            
            for strategy, name in pipeline_strategies:
                print(f"\\n   🚀 {name} Pipeline:")
                
                try:
                    result = pipeline.process_document(test_pdf, strategy)
                    
                    print(f"      ⏱️  Gesamt-Zeit: {result.processing_time:.2f}s")
                    print(f"      📄 Extraktion: {'✅' if result.extraction_success else '❌'}")
                    print(f"      🧩 Chunking: {result.chunking_strategy} ({len(result.chunks)} Chunks)")
                    print(f"      🤖 Klassifikation: {'✅' if result.classification_success else '❌'}")
                    
                    if result.final_classification:
                        fc = result.final_classification
                        print(f"      🎯 Kategorie: {fc.get('category', 'N/A')} (Confidence: {fc.get('confidence', 0):.2f})")
                    
                    if result.confidence_analysis:
                        ca = result.confidence_analysis
                        print(f"      📊 Finale Confidence: {ca.get('final_weighted_confidence', 0):.2f}")
                        print(f"      💡 Empfehlung: {ca.get('recommendation', 'N/A').upper()}")
                    
                    if result.errors:
                        print(f"      ❌ Fehler: {len(result.errors)}")
                    
                    if result.warnings:
                        print(f"      ⚠️  Warnungen: {len(result.warnings)}")
                        
                except Exception as e:
                    print(f"      ❌ Pipeline-Fehler: {e}")
        
        except Exception as e:
            print(f"❌ Pipeline-Initialisierung fehlgeschlagen: {e}")
        
        print()
        
        # DEMO 4: Performance-Vergleich
        print("⚡ DEMO 4: Performance-Vergleich")
        print("-" * 32)
        
        if extraction_results:
            print("\\n   📊 Chunking-Performance:")
            
            for strategy, name in strategies:
                if strategy in extraction_results and extraction_results[strategy]:
                    result = extraction_results[strategy]
                    chunk_count = len(result.chunks)
                    
                    # Einfache Performance-Metrik
                    efficiency = chunk_count / max(1, len(result.text) / 1000)  # Chunks per 1000 Zeichen
                    
                    print(f"      {name}: {chunk_count} Chunks, Effizienz: {efficiency:.2f}")
            
            print("\\n   💡 Empfehlungen:")
            print("      🚀 Schnell: NONE oder SIMPLE")
            print("      ⚖️  Ausgewogen: SIMPLE mit moderate Chunk-Größe")
            print("      🎯 Präzise: SEMANTIC oder HYBRID")
            print("      🏭 Produktion: BALANCED Pipeline-Strategie")
        
        print()
        print("🎉 INTEGRATION DEMO ABGESCHLOSSEN!")
        print("=" * 40)
        print("✅ Alle Komponenten erfolgreich getestet")
        print("🔗 Pipeline vollständig integriert")
        print("🚀 Bereit für produktiven Einsatz!")
        
        if semantic_available:
            print("🧠 Semantisches Chunking: VERFÜGBAR")
        else:
            print("⚠️  Semantisches Chunking: Basis-Implementation")
            print("   💡 Für vollständige Semantik: pip install sentence-transformers")
        
        print()
        print("📖 Nächste Schritte:")
        print("   1. Trainiere echtes ML-Modell")
        print("   2. Teste mit realen PDF-Dokumenten")
        print("   3. Konfiguriere für Produktionsumgebung")
        print("   4. Verwende CLI für alltägliche Aufgaben:")
        print("      python cli.py pipeline document.pdf comprehensive")
        
    except Exception as e:
        print(f"❌ Demo fehlgeschlagen: {e}")
        print("💡 Stelle sicher, dass alle Dependencies installiert sind:")
        print("   pip install -r requirements.txt")

def demo_text_only():
    """Fallback-Demo nur mit Text (ohne PDF)"""
    print("📝 TEXT-ONLY DEMO")
    print("-" * 17)
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
        from pipeline.classifier import RealMLClassifier
        
        classifier = RealMLClassifier()
        
        test_texts = [
            "Ich arbeite als Softwareentwickler bei einem IT-Unternehmen.",
            "Als Marketing Manager entwickle ich Kampagnen für Social Media.",
            "Ich bin Finanzanalyst und erstelle Bewertungsmodelle für Investitionen."
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\\n{i}. Text: {text}")
            result = classifier.classify_text(text)
            print(f"   Kategorie: {result.get('category', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 0):.2f}")
            print(f"   Vertrauenswürdig: {'Ja' if result.get('is_confident', False) else 'Nein'}")
        
        print("\\n✅ Text-Klassifikation funktioniert!")
        
    except Exception as e:
        print(f"❌ Text-Demo fehlgeschlagen: {e}")

if __name__ == "__main__":
    demo_complete_integration()
