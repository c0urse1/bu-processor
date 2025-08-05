#!/usr/bin/env python3
"""
🎯 BU PROCESSOR CLI - ERWEITERTE VERSION MIT SEMANTISCHEM CHUNKING
================================================================
Umfassendes CLI für PDF-Verarbeitung, semantisches Chunking und ML-Klassifikation
"""

import sys
import os

def main():
    """Hauptfunktion für erweiterte CLI"""
    
    print("🎯 BU Processor CLI - Erweitert mit Semantic Chunking")
    print("===================================================")
    
    if len(sys.argv) < 2:
        print("Verfügbare Kommandos:")
        print("  demo     - Klassifikations-Demo ausführen (Text + PDF + Chunking)")
        print("  pipeline - Integrierte End-to-End Pipeline testen")
        print("  pdf      - PDF-Verarbeitung demonstrieren")
        print("  chunks   - Chunking-Strategien testen")
        print("  classify - Datei oder Text klassifizieren")
        print("  batch    - Mehrere PDFs verarbeiten")
        print("  config   - Konfiguration anzeigen")
        print("  test     - Tests ausführen (wenn pytest installiert)")
        print("  pinecone - Pinecone Vector Database Demo")
        print("  enhanced - Enhanced Pipeline mit Pinecone testen")
        print("  search   - Vector Search in Pinecone Index")
        print("  🤖 CHATBOT INTEGRATION:")
        print("  chat     - Interactive Chatbot mit RAG")
        print("  ask      - Einzelne Frage an Chatbot")
        print("  chatdemo - Chatbot Integration Demo")
        print("  web      - Web Interface starten")
        print("  livedemo - Live Demo aller Features")
        print()
        print("Beispiele:")
        print("  python cli.py demo")
        print("  python cli.py pipeline tests/fixtures/sample.pdf comprehensive")
        print("  python cli.py pdf tests/fixtures/sample.pdf")
        print("  python cli.py chunks tests/fixtures/sample.pdf semantic")
        print("  python cli.py batch tests/fixtures/ balanced")
        print("  python cli.py classify 'Ich bin Softwareentwickler'")
        print("  python cli.py pinecone status")
        print("  python cli.py enhanced tests/fixtures/sample.pdf vector")
        print("  python cli.py search 'Berufsunfähigkeitsversicherung'")
        print("  🤖 CHATBOT:")
        print("  python cli.py chat")
        print("  python cli.py ask 'Was ist eine BU-Versicherung?'")
        print("  python cli.py web")
        print("  python cli.py livedemo")
        return
    
    command = sys.argv[1].lower()
    
    if command == "demo":
        print("🚀 Starte erweiterte Klassifikations-Demo...")
        print()
        try:
            # Import classifier demo
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
            from pipeline.classifier import demo_classifier
            demo_classifier()
        except ImportError as e:
            print(f"❌ Import Fehler: {e}")
            print("💡 Stelle sicher, dass alle Module erstellt wurden.")
        except Exception as e:
            print(f"❌ Demo Fehler: {e}")
            
    elif command == "pipeline":
        print("🔗 Integrierte End-to-End Pipeline...")
        if len(sys.argv) < 3:
            print("⚠️  Verwendung: python cli.py pipeline <pdf_datei> [strategie]")
            print("    Strategien: fast, balanced, comprehensive, semantic")
            return
        
        pdf_input = sys.argv[2]
        strategy = sys.argv[3] if len(sys.argv) > 3 else "balanced"
        
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
            from pipeline.integrated_pipeline import IntegratedPipeline, ProcessingStrategy
            from pathlib import Path
            
            # Strategy mapping
            strategy_map = {
                "fast": ProcessingStrategy.FAST,
                "balanced": ProcessingStrategy.BALANCED,
                "comprehensive": ProcessingStrategy.COMPREHENSIVE,
                "semantic": ProcessingStrategy.SEMANTIC_FOCUS
            }
            
            if strategy.lower() not in strategy_map:
                print(f"❌ Unbekannte Strategie: {strategy}")
                print("    Verfügbare Strategien: fast, balanced, comprehensive, semantic")
                return
            
            pipeline = IntegratedPipeline()
            input_path = Path(pdf_input)
            
            if input_path.is_file() and input_path.suffix.lower() == '.pdf':
                print(f"🔗 Starte {strategy.upper()} Pipeline für {input_path.name}...")
                
                result = pipeline.process_document(
                    input_path,
                    strategy=strategy_map[strategy.lower()]
                )
                
                print(f"\\n✅ Pipeline abgeschlossen in {result.processing_time:.2f}s")
                print(f"📁 Datei: {result.input_file}")
                print(f"📊 Pipeline-Version: {result.pipeline_version}")
                
                # Extraktion
                print(f"\\n📄 PDF-Extraktion:")
                print(f"   Status: {'✅ Erfolgreich' if result.extraction_success else '❌ Fehlgeschlagen'}")
                if result.extraction_success and result.extracted_content:
                    print(f"   Methode: {result.extraction_method}")
                    print(f"   Seiten: {result.extracted_content.page_count}")
                    print(f"   Text-Länge: {len(result.extracted_content.text)} Zeichen")
                
                # Chunking
                print(f"\\n🧩 Chunking:")
                print(f"   Status: {'✅ Aktiv' if result.chunking_success else '❌ Inaktiv'}")
                print(f"   Strategie: {result.chunking_strategy}")
                print(f"   Chunks erstellt: {len(result.chunks)}")
                
                if result.chunks:
                    avg_chunk_size = sum(len(c.text) for c in result.chunks) / len(result.chunks)
                    print(f"   Durchschn. Chunk-Größe: {avg_chunk_size:.0f} Zeichen")
                    print(f"   Chunk-Typen: {set(c.chunk_type for c in result.chunks)}")
                
                # Klassifikation
                print(f"\\n🤖 Klassifikation:")
                print(f"   Status: {'✅ Erfolgreich' if result.classification_success else '❌ Fehlgeschlagen'}")
                
                if result.final_classification:
                    fc = result.final_classification
                    print(f"   Kategorie: {fc.get('category', 'N/A')}")
                    print(f"   Confidence: {fc.get('confidence', 0):.2f}")
                    print(f"   Vertrauenswürdig: {'Ja' if fc.get('is_confident', False) else 'Nein'}")
                    print(f"   Input-Typ: {fc.get('input_type', 'N/A')}")
                    
                    if 'chunk_analysis' in fc:
                        ca = fc['chunk_analysis']
                        print(f"   Chunk-Analyse: {ca.get('processed_chunks', 0)} verarbeitet, {ca.get('high_confidence_chunks', 0)} hohe Confidence")
                
                # Semantische Analyse
                if result.semantic_analysis:
                    print(f"\\n🧠 Semantische Analyse:")
                    sa = result.semantic_analysis
                    print(f"   Enhanced: {'Ja' if sa.get('semantic_enhancement_applied', False) else 'Nein'}")
                    print(f"   Clusters: {sa.get('semantic_clusters_found', 'N/A')}")
                    print(f"   Kohärenz: {sa.get('cluster_coherence_average', 0):.2f}")
                
                # Qualitätsmetriken
                if result.quality_metrics:
                    print(f"\\n📈 Qualitätsmetriken:")
                    qm = result.quality_metrics
                    print(f"   Gesamt-Confidence: {qm.get('overall_confidence', 0):.2f}")
                    print(f"   Text-Qualität: {qm.get('text_avg_sentence_length', 0):.1f} Wörter/Satz")
                    print(f"   Erfolgsrate: {qm.get('processing_success_rate', 0):.1%}")
                    
                    if 'chunk_confidence_mean' in qm:
                        print(f"   Chunk-Confidence (Ø): {qm['chunk_confidence_mean']:.2f}")
                        print(f"   Hohe Confidence Chunks: {qm.get('high_confidence_chunks_ratio', 0):.1%}")
                
                # Finale Empfehlung
                if result.confidence_analysis:
                    print(f"\\n🎯 Finale Bewertung:")
                    ca = result.confidence_analysis
                    print(f"   Finale Confidence: {ca.get('final_weighted_confidence', 0):.2f}")
                    print(f"   Empfehlung: {ca.get('recommendation', 'N/A').upper()}")
                    print(f"   Confidence-Faktoren: {ca.get('confidence_factors_count', 0)}")
                
                # Fehler und Warnungen
                if result.errors:
                    print(f"\\n❌ Fehler ({len(result.errors)}):")
                    for error in result.errors:
                        print(f"   - {error}")
                
                if result.warnings:
                    print(f"\\n⚠️  Warnungen ({len(result.warnings)}):")
                    for warning in result.warnings:
                        print(f"   - {warning}")
                
                print(f"\\n🎉 Pipeline-Analyse abgeschlossen!")
                
            else:
                print(f"❌ Ungültiger PDF-Pfad: {pdf_input}")
                
        except Exception as e:
            print(f"❌ Pipeline fehlgeschlagen: {e}")
            
    elif command == "pdf":
        print("📄 PDF-Verarbeitung Demo...")
        if len(sys.argv) < 3:
            print("⚠️  Verwendung: python cli.py pdf <pdf_datei_oder_verzeichnis>")
            return
        
        pdf_input = sys.argv[2]
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
            from pipeline.pdf_extractor import EnhancedPDFExtractor
            from pathlib import Path
            
            extractor = EnhancedPDFExtractor()
            input_path = Path(pdf_input)
            
            if input_path.is_file() and input_path.suffix.lower() == '.pdf':
                # Einzelne PDF verarbeiten
                result = extractor.extract_text_from_pdf(input_path)
                print(f"✅ PDF erfolgreich extrahiert:")
                print(f"   Datei: {result.file_path}")
                print(f"   Seiten: {result.page_count}")
                print(f"   Methode: {result.extraction_method}")
                print(f"   Text-Länge: {len(result.text)} Zeichen")
                print(f"   Erste 200 Zeichen: {result.text[:200]}...")
                
            elif input_path.is_dir():
                # Verzeichnis verarbeiten
                results = extractor.extract_multiple_pdfs(input_path)
                print(f"✅ {len(results)} PDFs extrahiert aus {input_path}")
                for result in results:
                    print(f"   - {Path(result.file_path).name}: {result.page_count} Seiten")
            else:
                print(f"❌ Ungültiger Pfad oder keine PDF: {pdf_input}")
                
        except Exception as e:
            print(f"❌ PDF-Verarbeitung fehlgeschlagen: {e}")
    
    elif command == "chunks":
        print("🧩 Chunking-Strategien Demo...")
        if len(sys.argv) < 3:
            print("⚠️  Verwendung: python cli.py chunks <pdf_datei> [strategie]")
            print("    Strategien: none, simple, semantic, hybrid")
            return
        
        pdf_input = sys.argv[2]
        strategy = sys.argv[3] if len(sys.argv) > 3 else "simple"
        
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
            from pipeline.pdf_extractor import EnhancedPDFExtractor, ChunkingStrategy
            from pathlib import Path
            
            # Strategy mapping
            strategy_map = {
                "none": ChunkingStrategy.NONE,
                "simple": ChunkingStrategy.SIMPLE,
                "semantic": ChunkingStrategy.SEMANTIC,
                "hybrid": ChunkingStrategy.HYBRID
            }
            
            if strategy.lower() not in strategy_map:
                print(f"❌ Unbekannte Strategie: {strategy}")
                print("    Verfügbare Strategien: none, simple, semantic, hybrid")
                return
            
            extractor = EnhancedPDFExtractor(enable_chunking=True)
            input_path = Path(pdf_input)
            
            if input_path.is_file() and input_path.suffix.lower() == '.pdf':
                print(f"🧩 Teste Chunking-Strategie: {strategy.upper()}")
                
                result = extractor.extract_text_from_pdf(
                    input_path,
                    chunking_strategy=strategy_map[strategy.lower()],
                    max_chunk_size=800,
                    overlap_size=100
                )
                
                print(f"✅ Chunking abgeschlossen:")
                print(f"   Datei: {result.file_path}")
                print(f"   Strategie: {result.chunking_method}")
                print(f"   Chunks erstellt: {len(result.chunks)}")
                print(f"   Text-Länge: {len(result.text)} Zeichen")
                
                if result.chunks:
                    print(f"\\n📊 Chunk-Details:")
                    for i, chunk in enumerate(result.chunks[:3]):  # Erste 3 Chunks
                        print(f"   Chunk {i+1}: {len(chunk.text)} Zeichen, Score: {chunk.importance_score:.2f}")
                        print(f"   Typ: {chunk.chunk_type}")
                        print(f"   Preview: {chunk.text[:150]}...")
                        print()
                
                if result.semantic_clusters:
                    print(f"🧠 Semantic Clustering:")
                    clusters = result.semantic_clusters
                    print(f"   Clusters: {clusters.get('total_clusters', 'N/A')}")
                    print(f"   Enhanced: {clusters.get('enhancement_applied', False)}")
                
            else:
                print(f"❌ Ungültiger PDF-Pfad: {pdf_input}")
                
        except Exception as e:
            print(f"❌ Chunking-Demo fehlgeschlagen: {e}")
    
    elif command == "classify":
        print("🤖 Erweiterte Klassifikation...")
        if len(sys.argv) < 3:
            print("⚠️  Verwendung: python cli.py classify <text_oder_pdf_pfad> [chunk_strategy]")
            print("    Chunk-Strategien: none, simple, semantic, hybrid")
            return
        
        input_data = sys.argv[2]
        chunk_strategy = sys.argv[3] if len(sys.argv) > 3 else "simple"
        
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
            from pipeline.classifier import RealMLClassifier
            from pipeline.pdf_extractor import ChunkingStrategy
            from pathlib import Path
            
            strategy_map = {
                "none": ChunkingStrategy.NONE,
                "simple": ChunkingStrategy.SIMPLE,
                "semantic": ChunkingStrategy.SEMANTIC,
                "hybrid": ChunkingStrategy.HYBRID
            }
            
            classifier = RealMLClassifier()
            input_path = Path(input_data)
            
            # PDF-Klassifikation mit Chunking
            if input_path.exists() and input_path.suffix.lower() == '.pdf':
                print(f"📄 PDF-Klassifikation mit {chunk_strategy.upper()} Chunking...")
                
                result = classifier.classify_pdf(
                    input_path,
                    chunking_strategy=strategy_map.get(chunk_strategy.lower(), ChunkingStrategy.SIMPLE),
                    max_chunk_size=1000,
                    classify_chunks_individually=True if chunk_strategy.lower() != "none" else False
                )
                
                print(f"✅ Klassifikation abgeschlossen:")
                print(f"   Input-Typ: {result.get('input_type', 'unbekannt')}")
                
                if 'error' in result:
                    print(f"   ❌ Fehler: {result['error']}")
                else:
                    print(f"   Kategorie: {result.get('category', 'N/A')}")
                    print(f"   Confidence: {result.get('confidence', 0):.2f}")
                    print(f"   Sicher genug? {'Ja' if result.get('is_confident', False) else 'Nein'}")
                    print(f"   Chunking-Methode: {result.get('chunking_method', 'N/A')}")
                    
                    # Chunk-Analyse Details
                    if 'chunk_analysis' in result:
                        chunk_stats = result['chunk_analysis']
                        print(f"\\n📊 Chunk-Analyse:")
                        print(f"   Chunks verarbeitet: {chunk_stats.get('processed_chunks', 0)}")
                        print(f"   Durchschn. Confidence: {chunk_stats.get('average_chunk_confidence', 0):.2f}")
                        print(f"   Hohe Confidence: {chunk_stats.get('high_confidence_chunks', 0)} Chunks")
                        
                        if 'category_distribution' in chunk_stats:
                            print(f"   Kategorie-Verteilung: {chunk_stats['category_distribution']}")
            
            else:
                # Text-Klassifikation (unverändert)
                result = classifier.classify_text(str(input_data))
                
                print(f"✅ Text-Klassifikation abgeschlossen:")
                print(f"   Input-Typ: {result.get('input_type', 'text')}")
                print(f"   Kategorie: {result.get('category', 'N/A')}")
                print(f"   Confidence: {result.get('confidence', 0):.2f}")
                print(f"   Sicher genug? {'Ja' if result.get('is_confident', False) else 'Nein'}")
                    
        except Exception as e:
            print(f"❌ Erweiterte Klassifikation fehlgeschlagen: {e}")
    
    elif command == "batch":
        print("📁 Batch-Verarbeitung...")
        if len(sys.argv) < 3:
            print("⚠️  Verwendung: python cli.py batch <pdf_verzeichnis> [strategie]")
            print("    Strategien: fast, balanced, comprehensive, semantic")
            return
        
        pdf_directory = sys.argv[2]
        strategy = sys.argv[3] if len(sys.argv) > 3 else "balanced"
        
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
            from pipeline.integrated_pipeline import IntegratedPipeline, ProcessingStrategy
            from pathlib import Path
            
            strategy_map = {
                "fast": ProcessingStrategy.FAST,
                "balanced": ProcessingStrategy.BALANCED,
                "comprehensive": ProcessingStrategy.COMPREHENSIVE,
                "semantic": ProcessingStrategy.SEMANTIC_FOCUS
            }
            
            if strategy.lower() not in strategy_map:
                print(f"❌ Unbekannte Strategie: {strategy}")
                return
            
            pipeline = IntegratedPipeline()
            input_dir = Path(pdf_directory)
            
            if input_dir.is_dir():
                pdf_files = list(input_dir.glob("*.pdf"))
                
                if not pdf_files:
                    print(f"❌ Keine PDF-Dateien in {input_dir} gefunden")
                    return
                
                print(f"📁 Starte Batch-Verarbeitung für {len(pdf_files)} PDFs mit {strategy.upper()} Strategie...")
                
                results = pipeline.process_multiple_documents(
                    pdf_files,
                    strategy=strategy_map[strategy.lower()]
                )
                
                # Zusammenfassung
                successful = [r for r in results if len(r.errors) == 0]
                failed = [r for r in results if len(r.errors) > 0]
                
                print(f"\\n📊 Batch-Verarbeitung Zusammenfassung:")
                print(f"   Gesamt: {len(results)} Dateien")
                print(f"   ✅ Erfolgreich: {len(successful)}")
                print(f"   ❌ Fehlgeschlagen: {len(failed)}")
                
                if successful:
                    avg_time = sum(r.processing_time for r in successful) / len(successful)
                    avg_confidence = sum(r.confidence_analysis.get('final_weighted_confidence', 0) for r in successful if r.confidence_analysis) / len(successful)
                    print(f"   ⏱️  Durchschn. Zeit: {avg_time:.2f}s pro Datei")
                    print(f"   📈 Durchschn. Confidence: {avg_confidence:.2f}")
                
                # Detailierte Ergebnisse
                print(f"\\n📋 Detaillierte Ergebnisse:")
                for i, result in enumerate(results[:10]):  # Erste 10
                    file_name = Path(result.input_file).name
                    status = "✅" if len(result.errors) == 0 else "❌"
                    
                    if result.final_classification:
                        category = result.final_classification.get('category', 'N/A')
                        confidence = result.final_classification.get('confidence', 0)
                        print(f"   {status} {file_name}: Kategorie {category}, Confidence {confidence:.2f}")
                    else:
                        print(f"   {status} {file_name}: {result.errors[0] if result.errors else 'Unbekannter Fehler'}")
                
                if len(results) > 10:
                    print(f"   ... und {len(results) - 10} weitere")
                
            else:
                print(f"❌ Verzeichnis nicht gefunden: {pdf_directory}")
                
        except Exception as e:
            print(f"❌ Batch-Verarbeitung fehlgeschlagen: {e}")
    
    elif command == "config":
        print("📋 Aktuelle Konfiguration:")
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "core"))
            import config
            print(f"   Environment: {getattr(config, 'ENV', 'N/A')}")
            print(f"   ML Model Path: {getattr(config, 'ML_MODEL_PATH', 'N/A')}")
            print(f"   Confidence Threshold: {getattr(config, 'CONFIDENCE_THRESHOLD', 'N/A')}")
            print(f"   API: {getattr(config, 'API_HOST', 'N/A')}:{getattr(config, 'API_PORT', 'N/A')}")
            print(f"   GPU Enabled: {getattr(config, 'USE_GPU', 'N/A')}")
            print(f"   PDF Extraction Method: {getattr(config, 'PDF_EXTRACTION_METHOD', 'N/A')}")
            print(f"   Max PDF Size (MB): {getattr(config, 'MAX_PDF_SIZE_MB', 'N/A')}")
            print(f"   PDF Cache Enabled: {getattr(config, 'ENABLE_PDF_CACHE', 'N/A')}")
            print(f"   Semantic Chunking: {getattr(config, 'PDF_FALLBACK_CHAIN', 'N/A')}")
        except ImportError as e:
            print(f"❌ Config Import Fehler: {e}")
        
    elif command == "chat":
        print("🤖 Interactive Chatbot mit RAG...")
        
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
            from pipeline.chatbot_integration import ChatbotCLI
            
            chatbot_cli = ChatbotCLI()
            chatbot_cli.interactive_chat()
            
        except ImportError as e:
            print(f"❌ Chatbot nicht verfügbar: {e}")
            print("💡 Installiere: pip install openai rich")
        except Exception as e:
            print(f"❌ Chatbot fehlgeschlagen: {e}")
    
    elif command == "ask":
        print("🤖 Einzelne Frage an Chatbot...")
        
        if len(sys.argv) < 3:
            print("⚠️  Verwendung: python cli.py ask '<deine_frage>'")
            print("    Beispiel: python cli.py ask 'Was ist eine BU-Versicherung?'")
            return
        
        question = sys.argv[2]
        
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
            from pipeline.chatbot_integration import ChatbotCLI
            
            chatbot_cli = ChatbotCLI()
            chatbot_cli.single_query(question)
            
        except ImportError as e:
            print(f"❌ Chatbot nicht verfügbar: {e}")
            print("💡 Installiere: pip install openai rich")
        except Exception as e:
            print(f"❌ Frage fehlgeschlagen: {e}")
    
    elif command == "web":
        print("🌐 Web Interface starten...")
        
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
            from pipeline.chatbot_web_interface import run_web_server
            
            # Parse optional arguments
            host = "localhost"
            port = 8000
            
            if len(sys.argv) > 2:
                try:
                    port = int(sys.argv[2])
                except ValueError:
                    print(f"⚠️  Ungültiger Port: {sys.argv[2]}, verwende 8000")
            
            if len(sys.argv) > 3:
                host = sys.argv[3]
            
            print(f"🚀 Starte Web Server...")
            print(f"   🌐 URL: http://{host}:{port}")
            print(f"   📚 API: http://{host}:{port}/docs")
            print(f"   🛑 Stoppe mit Ctrl+C")
            
            run_web_server(host=host, port=port)
            
        except ImportError as e:
            print(f"❌ Web Interface nicht verfügbar: {e}")
            print("💡 Installiere: pip install fastapi uvicorn")
        except Exception as e:
            print(f"❌ Web Server fehlgeschlagen: {e}")
    
    elif command == "livedemo":
        print("🎬 Live Demo aller Chatbot Features...")
        
        try:
            # Import demo
            demo_file = os.path.join(os.path.dirname(__file__), "demo_chatbot_live.py")
            if os.path.exists(demo_file):
                os.system(f"python {demo_file}")
            else:
                print("❌ Live Demo Datei nicht gefunden")
                print("💡 Führe aus: python demo_chatbot_live.py")
                
        except Exception as e:
            print(f"❌ Live Demo fehlgeschlagen: {e}")
    
    elif command == "chatdemo":
        print("🤖 Chatbot Integration Demo...")
        
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
            from pipeline.chatbot_integration import demo_chatbot_integration
            
            demo_chatbot_integration()
            
        except ImportError as e:
            print(f"❌ Chatbot Demo nicht verfügbar: {e}")
            print("💡 Installiere: pip install openai rich")
        except Exception as e:
            print(f"❌ Chatbot Demo fehlgeschlagen: {e}")
    
    elif command == "test":
        print("🧪 Tests werden ausgeführt...")
        try:
            os.system("python -m pytest tests/ -v")
        except Exception as e:
            print(f"❌ Test Fehler: {e}")
            print("💡 Installiere pytest: pip install pytest")
    
    elif command == "pinecone":
        print("🌲 Pinecone Vector Database Demo...")
        
        action = sys.argv[2] if len(sys.argv) > 2 else "demo"
        
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
            from pipeline.pinecone_integration import demo_pinecone_integration, PineconePipeline, PineconeConfig
            
            if action == "demo":
                demo_pinecone_integration()
            elif action == "status":
                if not os.getenv("PINECONE_API_KEY"):
                    print("❌ PINECONE_API_KEY environment variable not set.")
                    return
                
                try:
                    config = PineconeConfig()
                    pipeline = PineconePipeline(config)
                    status = pipeline.get_pipeline_status()
                    
                    print(f"🌲 Pinecone Status:")
                    print(f"   Index: {status.get('index_stats', {}).get('index_name', 'N/A')}")
                    print(f"   Vectors: {status.get('index_stats', {}).get('total_vectors', 0):,}")
                    print(f"   Model: {status.get('embedding_model', 'N/A')}")
                    print(f"   Dimension: {status.get('embedding_dimension', 'N/A')}")
                    print(f"   Cache: {status.get('cache_stats', {}).get('cached_embeddings', 0)} embeddings")
                    
                except Exception as e:
                    print(f"❌ Pinecone Status Fehler: {e}")
            else:
                print(f"❌ Unbekannte Pinecone Aktion: {action}")
                print("   Verfügbare Aktionen: demo, status")
                
        except ImportError as e:
            print(f"❌ Pinecone nicht verfügbar: {e}")
            print("💡 Installiere mit: pip install pinecone-client sentence-transformers")
        except Exception as e:
            print(f"❌ Pinecone Demo fehlgeschlagen: {e}")
    
    elif command == "enhanced":
        print("🚀 Enhanced Pipeline mit Pinecone...")
        
        if len(sys.argv) < 3:
            print("⚠️  Verwendung: python cli.py enhanced <pdf_datei> [strategie]")
            print("    Strategien: fast, balanced, comprehensive, semantic, vector, vector_only")
            return
        
        pdf_input = sys.argv[2]
        strategy = sys.argv[3] if len(sys.argv) > 3 else "balanced"
        
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
            from pipeline.enhanced_integrated_pipeline import EnhancedIntegratedPipeline, EnhancedProcessingStrategy
            from pathlib import Path
            
            # Strategy mapping
            strategy_map = {
                "fast": EnhancedProcessingStrategy.FAST,
                "balanced": EnhancedProcessingStrategy.BALANCED,
                "comprehensive": EnhancedProcessingStrategy.COMPREHENSIVE,
                "semantic": EnhancedProcessingStrategy.SEMANTIC_FOCUS,
                "vector": EnhancedProcessingStrategy.VECTOR_ENHANCED,
                "vector_only": EnhancedProcessingStrategy.VECTOR_ONLY
            }
            
            if strategy.lower() not in strategy_map:
                print(f"❌ Unbekannte Strategie: {strategy}")
                print("    Verfügbare Strategien: fast, balanced, comprehensive, semantic, vector, vector_only")
                return
            
            # Initialisiere Enhanced Pipeline
            enable_pinecone = strategy.lower() in ["vector", "vector_only"]
            
            pipeline = EnhancedIntegratedPipeline(
                enable_pinecone=enable_pinecone,
                pinecone_config={
                    "index_name": "bu-processor-cli",
                    "embedding_model": "MULTILINGUAL_MINI"
                } if enable_pinecone else None
            )
            
            input_path = Path(pdf_input)
            
            if input_path.is_file() and input_path.suffix.lower() == '.pdf':
                print(f"🚀 Starte {strategy.upper()} Enhanced Pipeline für {input_path.name}...")
                
                result = pipeline.process_document(
                    input_path,
                    strategy=strategy_map[strategy.lower()]
                )
                
                print(f"\n✅ Enhanced Pipeline abgeschlossen in {result.processing_time:.2f}s")
                print(f"📁 Datei: {result.input_file}")
                print(f"📀 Pipeline-Version: {result.pipeline_version}")
                
                # Extraktion
                print(f"\n📄 PDF-Extraktion:")
                print(f"   Status: {'✅ Erfolgreich' if result.extraction_success else '❌ Fehlgeschlagen'}")
                if result.extraction_success and result.extracted_content:
                    print(f"   Methode: {result.extraction_method}")
                    print(f"   Seiten: {result.extracted_content.page_count}")
                    print(f"   Text-Länge: {len(result.extracted_content.text)} Zeichen")
                
                # Chunking
                print(f"\n🧩 Chunking:")
                print(f"   Status: {'✅ Aktiv' if result.chunking_success else '❌ Inaktiv'}")
                print(f"   Strategie: {result.chunking_strategy}")
                print(f"   Chunks erstellt: {len(result.chunks)}")
                
                # Klassifikation
                if result.final_classification:
                    print(f"\n🤖 Klassifikation:")
                    fc = result.final_classification
                    print(f"   Kategorie: {fc.get('category', 'N/A')}")
                    print(f"   Confidence: {fc.get('confidence', 0):.2f}")
                    print(f"   Vertrauenswürdig: {'Ja' if fc.get('is_confident', False) else 'Nein'}")
                
                # Pinecone Integration
                if result.pinecone_enabled:
                    print(f"\n🌲 Pinecone Integration:")
                    print(f"   Status: {'✅ Aktiv' if result.pinecone_enabled else '❌ Inaktiv'}")
                    
                    if result.pinecone_upload:
                        pu = result.pinecone_upload
                        print(f"   Uploads: {pu.get('uploaded', 0)} chunks")
                        print(f"   Namespace: {pu.get('namespace', 'N/A')}")
                        print(f"   Upload-Rate: {pu.get('pinecone_upload', {}).get('rate_per_second', 0):.1f}/s")
                    
                    if result.vector_search_results:
                        print(f"   Vector Searches: {len(result.vector_search_results)} queries")
                        best_search = max(result.vector_search_results, key=lambda x: x.get('best_score', 0))
                        print(f"   Best Search Score: {best_search.get('best_score', 0):.3f}")
                    
                    if result.similar_documents:
                        print(f"   Similar Documents: {len(result.similar_documents)} found")
                        best_sim = result.similar_documents[0] if result.similar_documents else None
                        if best_sim:
                            print(f"   Best Match: {best_sim.get('source_file', 'N/A')} (Score: {best_sim.get('similarity_score', 0):.3f})")
                
                # Embedding Stats
                if result.embedding_stats:
                    print(f"\n📊 Embedding Statistics:")
                    es = result.embedding_stats
                    print(f"   Embeddings: {es.get('total_embeddings_generated', 0)}")
                    print(f"   Upload Time: {es.get('upload_time', 0):.2f}s")
                    print(f"   Namespace: {es.get('upload_namespace', 'N/A')}")
                
                # Finale Bewertung
                if result.confidence_analysis:
                    print(f"\n🎯 Finale Bewertung:")
                    ca = result.confidence_analysis
                    print(f"   Finale Confidence: {ca.get('final_weighted_confidence', 0):.2f}")
                    print(f"   Empfehlung: {ca.get('recommendation', 'N/A').upper()}")
                    print(f"   Vector Enhanced: {'Ja' if ca.get('vector_enhanced', False) else 'Nein'}")
                
                # Fehler und Warnungen
                if result.errors:
                    print(f"\n❌ Fehler ({len(result.errors)}):")
                    for error in result.errors:
                        print(f"   - {error}")
                
                if result.warnings:
                    print(f"\n⚠️  Warnungen ({len(result.warnings)}):")
                    for warning in result.warnings:
                        print(f"   - {warning}")
                
                print(f"\n🎉 Enhanced Pipeline-Analyse abgeschlossen!")
                
            else:
                print(f"❌ Ungültiger PDF-Pfad: {pdf_input}")
                
        except ImportError as e:
            print(f"❌ Enhanced Pipeline nicht verfügbar: {e}")
            print("💡 Installiere Pinecone: pip install pinecone-client sentence-transformers")
        except Exception as e:
            print(f"❌ Enhanced Pipeline fehlgeschlagen: {e}")
    
    elif command == "search":
        print("🔍 Vector Search in Pinecone Index...")
        
        if len(sys.argv) < 3:
            print("⚠️  Verwendung: python cli.py search '<query>' [top_k] [namespace]")
            print("    Beispiel: python cli.py search 'Berufsunfähigkeitsversicherung' 5")
            return
        
        query = sys.argv[2]
        top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        namespace = sys.argv[4] if len(sys.argv) > 4 else ""
        
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
            from pipeline.enhanced_integrated_pipeline import EnhancedIntegratedPipeline
            
            if not os.getenv("PINECONE_API_KEY"):
                print("❌ PINECONE_API_KEY environment variable not set.")
                print("Set it with: export PINECONE_API_KEY='your-api-key'")
                return
            
            # Initialisiere Pipeline nur für Search
            pipeline = EnhancedIntegratedPipeline(
                enable_pinecone=True,
                pinecone_config={
                    "index_name": "bu-processor-cli",
                    "embedding_model": "MULTILINGUAL_MINI"
                }
            )
            
            print(f"🔍 Suche nach: '{query}'")
            print(f"   Top-K: {top_k}")
            print(f"   Namespace: {namespace or 'global'}")
            
            results = pipeline.search_in_index(
                query=query,
                top_k=top_k,
                namespace=namespace
            )
            
            if results:
                print(f"\n✅ {len(results)} Ergebnisse gefunden:")
                
                for i, result in enumerate(results, 1):
                    score = result.get("score", 0)
                    metadata = result.get("metadata", {})
                    preview = metadata.get("text_preview", "No preview")[:100]
                    source_file = metadata.get("source_file", "Unknown")
                    chunk_type = metadata.get("chunk_type", "Unknown")
                    
                    print(f"\n   {i}. Score: {score:.3f}")
                    print(f"      Quelle: {source_file}")
                    print(f"      Typ: {chunk_type}")
                    print(f"      Preview: {preview}...")
                    
                    if "heading_text" in metadata:
                        print(f"      Heading: {metadata['heading_text'][:50]}...")
            else:
                print(f"\n⚠️  Keine Ergebnisse für Query '{query}' gefunden.")
                print("   Tipp: Vergewissere dich, dass Dokumente im Index vorhanden sind.")
                print("   Verwende: python cli.py enhanced <pdf> vector_only zum Upload.")
                
        except ImportError as e:
            print(f"❌ Vector Search nicht verfügbar: {e}")
            print("💡 Installiere: pip install pinecone-client sentence-transformers")
        except Exception as e:
            print(f"❌ Vector Search fehlgeschlagen: {e}")
            
    elif command == "test":
        print("🧪 Tests werden ausgeführt...")
        try:
            os.system("python -m pytest tests/ -v")
        except Exception as e:
            print(f"❌ Test Fehler: {e}")
            print("💡 Installiere pytest: pip install pytest")
        
    else:
        print(f"❌ Unbekanntes Kommando: {command}")
        print("Verwende 'python cli.py' für verfügbare Kommandos.")
        print("\n🤖 Neue Chatbot-Features verfügbar!")
        print("   - chat: Interactive Chat mit RAG")
        print("   - ask: Einzelne Frage stellen")
        print("   - web: Web Interface (http://localhost:8000)")
        print("   - livedemo: Umfassende Feature-Demo")

if __name__ == "__main__":
    main()
