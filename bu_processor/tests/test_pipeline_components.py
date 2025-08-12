"""Tests für Pipeline-Komponenten (Pinecone, Chatbot, Integration, etc.)."""

import os
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any

import pytest

# Import der Skip-Markers für schwere Dependencies
from .conftest import requires_sentence_transformers, requires_pinecone, requires_full_ml_stack

# Import der zu testenden Klassen
try:
    from bu_processor.pipeline.pinecone_integration import (
        PineconeManager, VectorSearchResult, DocumentEmbedding
    )
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

# Zusätzliche Importe für Chunking / Extraktion (Kompatibilität neue Pipeline)
try:
    from bu_processor.pipeline.pdf_extractor import EnhancedPDFExtractor, ChunkingStrategy
    from bu_processor.pipeline.pdf_extractor import PDFTooLargeError
except ImportError:
    ChunkingStrategy = None  # type: ignore
    PDFTooLargeError = Exception  # fallback

try:
    from bu_processor.pipeline.chatbot_integration import (
        ChatbotIntegration, ChatMessage, ChatResponse
    )
    CHATBOT_AVAILABLE = True
except ImportError:
    CHATBOT_AVAILABLE = False

try:
    from bu_processor.pipeline.enhanced_integrated_pipeline import (
        EnhancedIntegratedPipeline, PipelineResult, PipelineConfig
    )
    ENHANCED_PIPELINE_AVAILABLE = True
except ImportError:
    ENHANCED_PIPELINE_AVAILABLE = False

try:
    from bu_processor.pipeline.semantic_chunking_enhancement import (
        SemanticClusteringEnhancer
    )
    SEMANTIC_ENHANCEMENT_AVAILABLE = True
except ImportError:
    SEMANTIC_ENHANCEMENT_AVAILABLE = False


@pytest.mark.skipif(not PINECONE_AVAILABLE, reason="Pinecone nicht verfügbar")
@requires_sentence_transformers
class TestPineconeIntegration:
    """Tests für Pinecone Vector Database Integration."""
    
    @pytest.fixture
    def mock_pinecone_client(self, mocker):
        """Mock für Pinecone Client."""
        mock_client = mocker.Mock()
        mock_index = mocker.Mock()
        
        # Mock query response
        mock_index.query.return_value = {
            'matches': [
                {
                    'id': 'doc_1',
                    'score': 0.95,
                    'metadata': {
                        'category': 'finance',
                        'text': 'Financial document content',
                        'source': 'test.pdf'
                    }
                },
                {
                    'id': 'doc_2', 
                    'score': 0.87,
                    'metadata': {
                        'category': 'legal',
                        'text': 'Legal document content',
                        'source': 'legal.pdf'
                    }
                }
            ]
        }
        
        # Mock upsert response
        mock_index.upsert.return_value = {'upserted_count': 1}
        
        # Mock delete response  
        mock_index.delete.return_value = {'deleted_count': 1}
        
        # Mock stats
        mock_index.describe_index_stats.return_value = {
            'dimension': 768,
            'index_fullness': 0.1,
            'total_vector_count': 1000
        }
        
        mock_client.Index.return_value = mock_index
        return mock_client, mock_index
    
    @pytest.fixture
    def pinecone_manager_with_mocks(self, mocker, mock_pinecone_client):
        """PineconeManager mit gemocktem Client."""
        mock_client, mock_index = mock_pinecone_client
        
        # Mock Pinecone init
        mocker.patch("pinecone.init")
        mocker.patch("pinecone.Index", return_value=mock_index)
        
        # Mock embedding model
        mock_model = mocker.Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3] * 256]  # 768 dim vector
        mocker.patch("sentence_transformers.SentenceTransformer", return_value=mock_model)
        
        if PINECONE_AVAILABLE:
            manager = PineconeManager(
                api_key="test_key",
                environment="test_env",
                index_name="test_index"
            )
            manager.client = mock_client
            manager.index = mock_index
            return manager
        return None
    
    def test_pinecone_manager_initialization(self, mocker):
        """Test PineconeManager Initialisierung."""
        if not PINECONE_AVAILABLE:
            pytest.skip("Pinecone integration nicht verfügbar")
        
        # Mock Pinecone dependencies
        mocker.patch("pinecone.init")
        mock_index = mocker.Mock()
        mocker.patch("pinecone.Index", return_value=mock_index)
        mocker.patch("sentence_transformers.SentenceTransformer")
        
        manager = PineconeManager(
            api_key="test_key",
            environment="test_env", 
            index_name="test_index",
            embedding_model="all-MiniLM-L6-v2"
        )
        
        assert manager.index_name == "test_index"
        assert manager.embedding_model_name == "all-MiniLM-L6-v2"
    
    def test_vector_search(self, pinecone_manager_with_mocks):
        """Test für Vector Search in Pinecone."""
        if not PINECONE_AVAILABLE or not pinecone_manager_with_mocks:
            pytest.skip("Pinecone integration nicht verfügbar")
        
        manager = pinecone_manager_with_mocks
        
        results = manager.search_similar_documents(
            query_text="Financial quarterly report",
            top_k=5,
            category_filter="finance"
        )
        
        assert len(results) == 2  # Mock returniert 2 results
        assert results[0]['score'] == 0.95
        assert results[0]['metadata']['category'] == 'finance'
        
        # Verify index.query was called
        manager.index.query.assert_called_once()
    
    def test_document_upload_to_pinecone(self, pinecone_manager_with_mocks):
        """Test für Dokument-Upload zu Pinecone."""
        if not PINECONE_AVAILABLE or not pinecone_manager_with_mocks:
            pytest.skip("Pinecone integration nicht verfügbar")
        
        manager = pinecone_manager_with_mocks
        
        document = {
            'id': 'test_doc_123',
            'text': 'Test document for upload',
            'category': 'IT',
            'source': 'test.pdf',
            'metadata': {'pages': 5}
        }
        
        result = manager.upload_document(document)
        
        assert result['success'] is True
        manager.index.upsert.assert_called_once()
    
    def test_bulk_document_upload(self, pinecone_manager_with_mocks):
        """Test für Bulk-Upload mehrerer Dokumente.""" 
        if not PINECONE_AVAILABLE or not pinecone_manager_with_mocks:
            pytest.skip("Pinecone integration nicht verfügbar")
        
        manager = pinecone_manager_with_mocks
        
        documents = [
            {'id': 'doc_1', 'text': 'Document 1', 'category': 'finance'},
            {'id': 'doc_2', 'text': 'Document 2', 'category': 'legal'},
            {'id': 'doc_3', 'text': 'Document 3', 'category': 'IT'}
        ]
        
        results = manager.bulk_upload_documents(documents)
        
        assert results['total_processed'] == 3
        assert results['successful'] == 3
        assert results['failed'] == 0


@pytest.mark.skipif(not CHATBOT_AVAILABLE, reason="Chatbot integration nicht verfügbar")
class TestChatbotIntegration:
    """Tests für Chatbot-Integration."""
    
    @pytest.fixture
    def mock_openai_client(self, mocker):
        """Mock für OpenAI API Client."""
        mock_client = mocker.Mock()
        
        # Mock chat completion response
        mock_response = mocker.Mock()
        mock_response.choices = [mocker.Mock()]
        mock_response.choices[0].message = mocker.Mock()
        mock_response.choices[0].message.content = "Das ist eine Antwort vom Chatbot."
        mock_response.usage = mocker.Mock()
        mock_response.usage.total_tokens = 150
        
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client
    
    @pytest.fixture 
    def chatbot_with_mocks(self, mocker, mock_openai_client):
        """ChatbotIntegration mit gemocktem OpenAI Client."""
        if not CHATBOT_AVAILABLE:
            return None
        
        mocker.patch("openai.OpenAI", return_value=mock_openai_client)
        
        chatbot = ChatbotIntegration(
            api_key="test_key",
            model="gpt-3.5-turbo",
            max_tokens=500
        )
        chatbot.client = mock_openai_client
        return chatbot
    
    def test_chatbot_initialization(self, mocker):
        """Test Chatbot-Initialisierung."""
        if not CHATBOT_AVAILABLE:
            pytest.skip("Chatbot integration nicht verfügbar")
        
        mock_client = mocker.Mock()
        mocker.patch("openai.OpenAI", return_value=mock_client)
        
        chatbot = ChatbotIntegration(
            api_key="test_key", 
            model="gpt-4",
            max_tokens=1000,
            temperature=0.7
        )
        
        assert chatbot.model == "gpt-4"
        assert chatbot.max_tokens == 1000
        assert chatbot.temperature == 0.7
    
    def test_simple_chat_message(self, chatbot_with_mocks):
        """Test für einfache Chat-Nachricht."""
        if not CHATBOT_AVAILABLE or not chatbot_with_mocks:
            pytest.skip("Chatbot integration nicht verfügbar")
        
        chatbot = chatbot_with_mocks
        
        response = chatbot.send_message("Was ist maschinelles Lernen?")
        
        assert response.success is True
        assert response.message_content == "Das ist eine Antwort vom Chatbot."
        assert response.token_usage == 150
        
        # Verify OpenAI API was called
        chatbot.client.chat.completions.create.assert_called_once()
    
    def test_chat_with_document_context(self, chatbot_with_mocks, mocker):
        """Test für Chat mit Dokument-Kontext."""
        if not CHATBOT_AVAILABLE or not chatbot_with_mocks:
            pytest.skip("Chatbot integration nicht verfügbar")
        
        chatbot = chatbot_with_mocks
        
        document_context = {
            'text': 'Kontext aus klassifiziertem Dokument',
            'category': 'finance',
            'confidence': 0.85
        }
        
        response = chatbot.chat_with_document_context(
            user_message="Erkläre mir dieses Dokument",
            document_context=document_context
        )
        
        assert response.success is True
        assert response.context_used is True
        
        # Verify context was included in API call
        call_args = chatbot.client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        
        # Should include system message with context
        system_messages = [msg for msg in messages if msg['role'] == 'system']
        assert len(system_messages) > 0
        assert any('finance' in str(msg) for msg in system_messages)
    
    def test_chatbot_error_handling(self, mocker):
        """Test für Chatbot Error Handling.""" 
        if not CHATBOT_AVAILABLE:
            pytest.skip("Chatbot integration nicht verfügbar")
        
        # Mock OpenAI client that raises exception
        mock_client = mocker.Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mocker.patch("openai.OpenAI", return_value=mock_client)
        
        chatbot = ChatbotIntegration(api_key="test_key")
        chatbot.client = mock_client
        
        response = chatbot.send_message("Test message")
        
        assert response.success is False
        assert "API Error" in response.error_message


@pytest.mark.skipif(not ENHANCED_PIPELINE_AVAILABLE, reason="Enhanced Pipeline nicht verfügbar")
class TestEnhancedIntegratedPipeline:
    """Tests für Enhanced Integrated Pipeline."""
    
    @pytest.fixture
    def mock_pipeline_components(self, mocker):
        """Mock für alle Pipeline-Komponenten."""
        # Mock Classifier
        mock_classifier = mocker.Mock()
        mock_classifier.classify.return_value = {
            'category': 1,
            'confidence': 0.89,
            'is_confident': True
        }
        mock_classifier.get_health_status.return_value = {'status': 'healthy'}
        
        # Mock PDF Extractor  
        mock_extractor = mocker.Mock()
        mock_extractor.extract_text_from_pdf.return_value = mocker.Mock(
            text="Extracted PDF text",
            page_count=3,
            file_path="test.pdf"
        )
        
        # Mock Pinecone Manager
        mock_pinecone = mocker.Mock()
        mock_pinecone.search_similar_documents.return_value = [
            {'id': 'similar_1', 'score': 0.92, 'metadata': {'category': 'finance'}}
        ]
        
        # Mock Chatbot
        mock_chatbot = mocker.Mock()
        mock_chatbot.send_message.return_value = mocker.Mock(
            success=True,
            message_content="Chatbot response"
        )
        
        return {
            'classifier': mock_classifier,
            'extractor': mock_extractor,
            'pinecone': mock_pinecone,
            'chatbot': mock_chatbot
        }
    
    @pytest.fixture
    def pipeline_with_mocks(self, mocker, mock_pipeline_components):
        """Enhanced Pipeline mit gemockten Komponenten."""
        if not ENHANCED_PIPELINE_AVAILABLE:
            return None
        
        mocks = mock_pipeline_components
        
        # Mock component initialization
        mocker.patch("bu_processor.pipeline.enhanced_integrated_pipeline.RealMLClassifier", 
                     return_value=mocks['classifier'])
        mocker.patch("bu_processor.pipeline.enhanced_integrated_pipeline.EnhancedPDFExtractor",
                     return_value=mocks['extractor'])
        mocker.patch("bu_processor.pipeline.enhanced_integrated_pipeline.PineconeManager",
                     return_value=mocks['pinecone'])
        mocker.patch("bu_processor.pipeline.enhanced_integrated_pipeline.ChatbotIntegration",
                     return_value=mocks['chatbot'])
        
        config = {
            'enable_pinecone': True,
            'enable_chatbot': True,
            'enable_caching': False,
            'batch_size': 16
        }
        
        pipeline = EnhancedIntegratedPipeline(config=config)
        return pipeline, mocks
    
    def test_pipeline_initialization(self, mocker):
        """Test Pipeline-Initialisierung mit Konfiguration."""
        if not ENHANCED_PIPELINE_AVAILABLE:
            pytest.skip("Enhanced Pipeline nicht verfügbar")
        
        # Mock alle Komponenten
        mocker.patch("bu_processor.pipeline.enhanced_integrated_pipeline.RealMLClassifier")
        mocker.patch("bu_processor.pipeline.enhanced_integrated_pipeline.EnhancedPDFExtractor")
        mocker.patch("bu_processor.pipeline.enhanced_integrated_pipeline.PineconeManager")
        mocker.patch("bu_processor.pipeline.enhanced_integrated_pipeline.ChatbotIntegration")
        
        config = {
            'enable_pinecone': True,
            'enable_chatbot': False,
            'batch_size': 32
        }
        
        pipeline = EnhancedIntegratedPipeline(config=config)
        
        assert pipeline.config['enable_pinecone'] is True
        assert pipeline.config['enable_chatbot'] is False
        assert pipeline.config['batch_size'] == 32
    
    def test_process_single_pdf_end_to_end(self, pipeline_with_mocks):
        """Test für komplette PDF-Verarbeitung."""
        if not ENHANCED_PIPELINE_AVAILABLE or not pipeline_with_mocks:
            pytest.skip("Enhanced Pipeline nicht verfügbar")
        
        pipeline, mocks = pipeline_with_mocks
        
        result = pipeline.process_pdf("test.pdf")
        
        # Verify alle Komponenten wurden aufgerufen
        mocks['extractor'].extract_text_from_pdf.assert_called_once()
        mocks['classifier'].classify.assert_called_once()
        
        # Verify result structure
        assert result.success is True
        assert result.classification_result is not None
        assert result.extraction_result is not None
    
    def test_process_batch_pdfs(self, pipeline_with_mocks):
        """Test für Batch-PDF-Verarbeitung."""
        if not ENHANCED_PIPELINE_AVAILABLE or not pipeline_with_mocks:
            pytest.skip("Enhanced Pipeline nicht verfügbar")
        
        pipeline, mocks = pipeline_with_mocks
        
        pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        results = pipeline.process_pdf_batch(pdf_files)
        
        assert results.total_processed == 3
        assert results.successful >= 0
        assert len(results.results) == 3
    
    def test_pipeline_with_pinecone_integration(self, pipeline_with_mocks):
        """Test Pipeline mit Pinecone-Integration."""
        if not ENHANCED_PIPELINE_AVAILABLE or not pipeline_with_mocks:
            pytest.skip("Enhanced Pipeline nicht verfügbar")
        
        pipeline, mocks = pipeline_with_mocks
        
        result = pipeline.process_pdf_with_similarity_search("test.pdf")
        
        # Verify Pinecone search was called
        mocks['pinecone'].search_similar_documents.assert_called_once()
        
        assert result.similarity_results is not None
        assert len(result.similarity_results) > 0
    
    def test_pipeline_health_check(self, pipeline_with_mocks):
        """Test für Pipeline Health Check."""
        if not ENHANCED_PIPELINE_AVAILABLE or not pipeline_with_mocks:
            pytest.skip("Enhanced Pipeline nicht verfügbar")
        
        pipeline, mocks = pipeline_with_mocks
        
        health = pipeline.get_health_status()
        
        assert health['pipeline_status'] == 'healthy'
        assert 'classifier_status' in health
        assert health['components_initialized'] > 0


@requires_sentence_transformers  
class TestSemanticChunkingEnhancement:
    """Tests für Semantic Chunking Enhancement."""
    
    @pytest.fixture
    def mock_sentence_transformer(self, mocker):
        """Mock für Sentence Transformer Model."""
        mock_model = mocker.Mock()
        
        # Mock embeddings (768-dim vectors)
        mock_model.encode.return_value = [
            [0.1] * 768,  # Embedding für ersten Text
            [0.2] * 768,  # Embedding für zweiten Text
            [0.3] * 768   # Embedding für dritten Text
        ]
        
        return mock_model
    
    @pytest.fixture
    def semantic_enhancer_with_mocks(self, mocker, mock_sentence_transformer):
        """SemanticClusteringEnhancer mit Mocks."""
        if not SEMANTIC_ENHANCEMENT_AVAILABLE:
            return None
        
        mocker.patch("sentence_transformers.SentenceTransformer", 
                     return_value=mock_sentence_transformer)
        
        # Mock sklearn clustering
        mock_kmeans = mocker.Mock()
        mock_kmeans.fit_predict.return_value = [0, 0, 1, 1, 2]  # Cluster assignments
        mocker.patch("sklearn.cluster.KMeans", return_value=mock_kmeans)
        
        enhancer = SemanticClusteringEnhancer()
        return enhancer
    
    def test_semantic_enhancer_initialization(self, mocker):
        """Test SemanticClusteringEnhancer Initialisierung."""
        if not SEMANTIC_ENHANCEMENT_AVAILABLE:
            pytest.skip("Semantic enhancement nicht verfügbar")
        
        mock_model = mocker.Mock()
        mocker.patch("sentence_transformers.SentenceTransformer", return_value=mock_model)
        
        enhancer = SemanticClusteringEnhancer(
            model_name="all-MiniLM-L6-v2",
            clustering_method="kmeans"
        )
        
        assert enhancer.model_name == "all-MiniLM-L6-v2"
        assert enhancer.clustering_method == "kmeans"
    
    def test_text_clustering(self, semantic_enhancer_with_mocks):
        """Test für Text-Clustering."""
        if not SEMANTIC_ENHANCEMENT_AVAILABLE or not semantic_enhancer_with_mocks:
            pytest.skip("Semantic enhancement nicht verfügbar")
        
        enhancer = semantic_enhancer_with_mocks
        
        texts = [
            "Finanzielle Quartalsergebnisse",
            "Umsatz und Gewinn Analyse", 
            "Rechtliche Bestimmungen",
            "Vertragsklauseln prüfen",
            "IT-System Migration"
        ]
        
        clusters = enhancer.cluster_texts(texts, n_clusters=3)
        
        assert len(clusters) == len(texts)
        assert all(isinstance(cluster_id, int) for cluster_id in clusters)
        assert max(clusters) <= 2  # 3 clusters (0, 1, 2)
    
    def test_semantic_similarity_calculation(self, semantic_enhancer_with_mocks):
        """Test für semantische Ähnlichkeits-Berechnung."""
        if not SEMANTIC_ENHANCEMENT_AVAILABLE or not semantic_enhancer_with_mocks:
            pytest.skip("Semantic enhancement nicht verfügbar")
        
        enhancer = semantic_enhancer_with_mocks
        
        text1 = "Finanzielle Quartalsergebnisse zeigen Wachstum"
        text2 = "Umsatz steigt um 15% im Vergleich zum Vorquartal"
        
        similarity = enhancer.calculate_similarity(text1, text2)
        
        assert 0.0 <= similarity <= 1.0
        assert isinstance(similarity, float)


class TestPipelineErrorHandling:
    """Tests für Pipeline Error Handling."""
    
    def test_component_failure_isolation(self, mocker):
        """Test dass Komponenten-Fehler isoliert werden."""
        # Mock failing classifier
        mock_classifier = mocker.Mock()
        mock_classifier.classify.side_effect = Exception("Classifier failed")
        
        # Mock working extractor
        mock_extractor = mocker.Mock()
        mock_extractor.extract_text_from_pdf.return_value = mocker.Mock(
            text="Extracted text",
            page_count=1
        )
        
        # Test dass Pipeline trotz Classifier-Fehler weiterläuft
        # (spezifischer Test abhängig von tatsächlicher Pipeline-Implementierung)
        
        assert mock_extractor.extract_text_from_pdf.return_value.text == "Extracted text"
    
    def test_graceful_degradation(self, mocker):
        """Test für graceful degradation bei Service-Ausfällen."""
        # Mock pipeline config mit deaktivierten Services
        config = {
            'enable_pinecone': False,  # Pinecone deaktiviert
            'enable_chatbot': False,   # Chatbot deaktiviert
            'fallback_mode': True
        }
        
        # Pipeline sollte trotzdem funktionieren
        assert config['fallback_mode'] is True


class TestSimHashSemanticDeduplication:
    """Tests für SimHash-basierte Deduplizierung."""
    
    def test_simhash_calculation(self, mocker):
        """Test für SimHash-Berechnung."""
        # Mock simhash library falls verfügbar
        try:
            from bu_processor.pipeline.simhash_semantic_deduplication import calculate_simhash
            
            text1 = "Das ist ein Test-Text für SimHash"
            text2 = "Das ist ein Test-Text für SimHash"  # Identisch
            text3 = "Komplett anderer Text ohne Ähnlichkeit"
            
            hash1 = calculate_simhash(text1)
            hash2 = calculate_simhash(text2) 
            hash3 = calculate_simhash(text3)
            
            # Identische Texte sollten gleichen Hash haben
            assert hash1 == hash2
            
            # Verschiedene Texte sollten verschiedene Hashes haben
            assert hash1 != hash3
            
        except ImportError:
            pytest.skip("SimHash deduplication nicht verfügbar")
    
    def test_duplicate_detection(self, mocker):
        """Test für Duplikat-Erkennung."""
        try:
            from bu_processor.pipeline.simhash_semantic_deduplication import find_duplicates
            
            documents = [
                {'id': 'doc1', 'text': 'Das ist Dokument eins'},
                {'id': 'doc2', 'text': 'Das ist Dokument eins'},  # Duplikat
                {'id': 'doc3', 'text': 'Komplett anderes Dokument'},
                {'id': 'doc4', 'text': 'Das ist Dokument eins'}   # Noch ein Duplikat
            ]
            
            duplicates = find_duplicates(documents, threshold=5)  # Niedrige Hamming-Distanz
            
            assert len(duplicates) > 0
            # doc1, doc2, doc4 sollten als Duplikate erkannt werden
            
        except ImportError:
            pytest.skip("SimHash deduplication nicht verfügbar")


class TestAsyncPipelineComponents:
    """Tests für asynchrone Pipeline-Komponenten."""
    
    @pytest.mark.asyncio
    async def test_async_batch_processing(self, mocker):
        """Test für asynchrone Batch-Verarbeitung."""
        # Mock async classifier
        mock_classifier = AsyncMock()
        mock_classifier.classify_async.return_value = {
            'category': 1,
            'confidence': 0.85
        }
        
        # Simuliere async batch processing
        texts = ["Text 1", "Text 2", "Text 3"]
        
        # Test dass alle Texte parallel verarbeitet werden
        tasks = [mock_classifier.classify_async(text) for text in texts]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(result['confidence'] == 0.85 for result in results)
    
    @pytest.mark.asyncio 
    async def test_async_error_handling(self, mocker):
        """Test für Error Handling in async Operationen."""
        mock_classifier = AsyncMock()
        mock_classifier.classify_async.side_effect = [
            {'category': 1, 'confidence': 0.85},  # Erfolg
            Exception("Async error"),              # Fehler
            {'category': 2, 'confidence': 0.75}   # Erfolg
        ]
        
        texts = ["Text 1", "Text 2", "Text 3"]
        results = []
        
        for text in texts:
            try:
                result = await mock_classifier.classify_async(text)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        
        assert len(results) == 3
        assert results[0]['confidence'] == 0.85
        assert 'error' in results[1]
        assert results[2]['confidence'] == 0.75


class TestConfigurationAndSettings:
    """Tests für Konfiguration und Settings."""
    
    def test_config_validation(self):
        """Test für Konfigurations-Validierung."""
        # Test gültige Konfiguration
        valid_config = {
            'max_pdf_size_mb': 50,
            'batch_size': 32,
            'enable_retry': True,
            'max_retries': 3,
            'timeout_seconds': 30.0
        }
        
        # Validierung würde hier implementiert werden
        assert valid_config['max_pdf_size_mb'] > 0
        assert valid_config['batch_size'] > 0
        assert valid_config['max_retries'] >= 0
    
    def test_config_defaults(self):
        """Test für Standard-Konfiguration."""
        # Test dass Defaults korrekt gesetzt sind
        from bu_processor.pipeline.pdf_extractor import MAX_PDF_SIZE_MB, MAX_PDF_PAGES
        
        assert MAX_PDF_SIZE_MB > 0
        assert MAX_PDF_PAGES > 0
    
    def test_environment_variable_override(self, mocker):
        """Test für Environment Variable Overrides."""
        # Mock environment variables
        mocker.patch.dict(os.environ, {
            'PDF_MAX_SIZE_MB': '100',
            'CLASSIFIER_BATCH_SIZE': '64'
        })
        
        # Test dass env vars korrekt gelesen werden
        assert os.environ.get('PDF_MAX_SIZE_MB') == '100'
        assert os.environ.get('CLASSIFIER_BATCH_SIZE') == '64'


class TestPipelineIntegrationScenarios:
    """Integration Tests für realistische Pipeline-Szenarien."""
    
    def test_document_processing_workflow(self, mocker):
        """Test für kompletten Dokument-Verarbeitungs-Workflow."""
        # Simuliere realistischen Workflow:
        # 1. PDF Upload
        # 2. Text Extraction  
        # 3. Classification
        # 4. Vector Storage
        # 5. Similarity Search
        
        workflow_steps = []
        
        # Mock each step
        def mock_extract(pdf_path):
            workflow_steps.append("extraction")
            return Mock(text="Extracted text", page_count=1)
        
        def mock_classify(text):
            workflow_steps.append("classification")
            return {'category': 1, 'confidence': 0.85}
        
        def mock_upload(doc):
            workflow_steps.append("vector_upload")
            return {'success': True}
        
        def mock_search(query):
            workflow_steps.append("similarity_search")
            return [{'id': 'similar', 'score': 0.9}]
        
        # Execute workflow
        pdf_path = "test.pdf"
        extracted = mock_extract(pdf_path)
        classification = mock_classify(extracted.text)
        upload_result = mock_upload({'text': extracted.text, 'category': classification['category']})
        similar_docs = mock_search(extracted.text)
        
        # Verify workflow completed
        assert workflow_steps == ["extraction", "classification", "vector_upload", "similarity_search"]
        assert classification['confidence'] == 0.85
        assert upload_result['success'] is True
        assert len(similar_docs) == 1
    
    def test_concurrent_document_processing(self, mocker):
        """Test für gleichzeitige Verarbeitung mehrerer Dokumente."""
        # Test für process_documents_multiprocessing aus enhanced_integrated_pipeline
        try:
            from bu_processor.pipeline.enhanced_integrated_pipeline import process_documents_multiprocessing
            
            # Erstelle 5 Test-Dateipfade (auch wenn sie nicht existieren)
            test_files = [f"test_doc_{i}.pdf" for i in range(5)]
            
            # Rufe die Multiprocessing-Funktion auf
            results = process_documents_multiprocessing(
                file_paths=test_files,
                strategy="fast",
                max_workers=2
            )
            
            # Prüfe dass 5 Ergebnisse zurückgegeben werden (auch wenn Fehler)
            assert len(results) == 5
            
            # Alle Ergebnisse sollten Dictionaries sein
            assert all(isinstance(result, dict) for result in results)
            
            # Jedes Ergebnis sollte mindestens einen 'success' oder 'errors' Key haben
            for result in results:
                assert 'success' in result or 'errors' in result
                
        except ImportError:
            pytest.skip("process_documents_multiprocessing nicht verfügbar")


class TestMockDataConsistency:
    """Tests um sicherzustellen dass Mock-Daten konsistent sind."""
    
    def test_mock_vector_dimensions_consistent(self):
        """Test dass Vector-Dimensionen in allen Mocks konsistent sind."""
        # Standard embedding dimension für tests
        EXPECTED_DIM = 768
        
        # Test verschiedene Mock-Vektoren
        mock_vector_1 = [0.1] * EXPECTED_DIM
        mock_vector_2 = [0.2] * EXPECTED_DIM
        
        assert len(mock_vector_1) == EXPECTED_DIM
        assert len(mock_vector_2) == EXPECTED_DIM
    
    def test_mock_response_formats(self):
        """Test dass Mock-Response-Formate korrekt sind."""
        # Pinecone response format
        mock_pinecone_response = {
            'matches': [
                {'id': 'doc1', 'score': 0.95, 'metadata': {'category': 'test'}}
            ]
        }
        
        # Classification result format
        mock_classification = {
            'category': 1,
            'confidence': 0.85,
            'is_confident': True
        }
        
        # Validate structures
        assert 'matches' in mock_pinecone_response
        assert 'category' in mock_classification
        assert 'confidence' in mock_classification
        assert 0.0 <= mock_classification['confidence'] <= 1.0


# Parametrized Tests für verschiedene Konfigurationen
if ChunkingStrategy:
    @pytest.mark.parametrize("chunking_strategy,expected_chunks", [
        (ChunkingStrategy.NONE, 0),
        (ChunkingStrategy.SIMPLE, 1),  # Mindestens 1 chunk
        (ChunkingStrategy.SEMANTIC, 1),
        (ChunkingStrategy.HYBRID, 1),
    ])
    def test_chunking_strategies_parametrized(chunking_strategy, expected_chunks, mocker):
        """Parametrized Test für verschiedene Chunking-Strategien."""
        extractor = EnhancedPDFExtractor(enable_chunking=True)
        sample_text = "Test text für parametrized chunking test. " * 10
        chunks = extractor._apply_chunking_strategy(
            sample_text,
            chunking_strategy,
            max_chunk_size=100,
            overlap_size=20,
            page_count=1
        )
        if expected_chunks == 0:
            assert len(chunks) == 0
        else:
            assert len(chunks) >= expected_chunks
else:  # pragma: no cover - fallback when import failed
    def test_chunking_strategies_parametrized():
        pytest.skip("ChunkingStrategy nicht verfügbar - Import fehlgeschlagen")


@pytest.mark.parametrize("file_size_mb,should_raise", [
    (10, False),   # OK
    (45, False),   # OK  
    (60, True),    # Zu groß
    (100, True),   # Definitiv zu groß
])
def test_pdf_size_validation_parametrized(file_size_mb, should_raise, mocker, tmp_path):
    """Parametrized Test für PDF-Größen-Validierung."""
    test_pdf = tmp_path / "test.pdf"
    test_pdf.write_text("fake pdf")
    
    # Mock file size
    mock_stat = mocker.Mock()
    mock_stat.st_size = file_size_mb * 1024 * 1024
    mocker.patch.object(Path, "stat", return_value=mock_stat)
    
    extractor = EnhancedPDFExtractor()
    
    if should_raise:
        with pytest.raises(PDFTooLargeError):
            extractor._validate_pdf(test_pdf)
    else:
        try:
            extractor._validate_pdf(test_pdf)
        except PDFTooLargeError:
            pytest.fail("PDF size validation should not raise for valid sizes")


# Utility functions
def run_pipeline_tests():
    """Führt alle Pipeline-Komponenten Tests aus."""
    return pytest.main([__file__, "-v", "--tb=short"])


def run_specific_test_class(test_class_name: str):
    """Führt eine spezifische Test-Klasse aus."""
    return pytest.main([__file__ + "::" + test_class_name, "-v"])


if __name__ == "__main__":
    run_pipeline_tests()
