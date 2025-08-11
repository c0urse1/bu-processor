#!/usr/bin/env python3
"""
🌳 PHASE 4: HIERARCHICAL & SEMANTIC CHUNKING ENHANCEMENT
========================================================

Erweitert das hierarchische Chunking-System um semantisches Clustering
mit SentenceTransformers für intelligentere Dokumentsegmentierung.

NEUE FEATURES:
- SentenceTransformer-basiertes semantisches Clustering
- Adaptive Cluster-Parameter basierend auf Content-Type
- Semantische Ähnlichkeitsmetriken zwischen Chunks
- Cluster-basierte Chunk-Optimierung
- Cross-linguales Embedding (DE/EN) für BU-Dokumente
"""

# Standard library imports
import hashlib
import time
from functools import lru_cache
from typing import Dict, List, Any, Optional, Tuple, Union, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum  # retained for backwards compatibility references
from .content_types import ContentType  # unified enum
try:  # Optional import for type hints
    from .simhash_semantic_deduplication import HierarchicalChunk  # type: ignore
except Exception:  # pragma: no cover
    class HierarchicalChunk:  # minimal placeholder for type checkers
        pass
import os

# Third-party imports
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.manifold import TSNE
    import numpy as np
    SEMANTIC_DEPS_AVAILABLE = True
except ImportError as e:
    SEMANTIC_DEPS_AVAILABLE = False
    # Mock classes for type hints when dependencies not available
    SentenceTransformer = None
    np = None

# Config import mit zentraler Konfiguration
try:
    from ..core.config import (
        settings, SENTENCE_TRANSFORMER_MODEL, LOG_LEVEL, SEMANTIC_CONFIG
    )
    # Logger Setup aus config
    import structlog
    import logging
    
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
    logger = structlog.get_logger("semantic_clustering")
    
except ImportError:
    # Fallback values
    SENTENCE_TRANSFORMER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    LOG_LEVEL = "INFO"
    
    # Fallback Konfiguration falls zentrale Config nicht verfügbar
    SEMANTIC_CONFIG = {
        'models': {
            'multilingual': 'paraphrase-multilingual-MiniLM-L12-v2',
            'german_legal': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            'fast_embedding': 'paraphrase-MiniLM-L6-v2'
        },
        'batch_processing': {
            'default_batch_size': 32,
            'large_batch_size': 64,
            'small_batch_size': 16,
            'max_chunks_for_batching': 100
        },
        'clustering': {
            'legal_text': {
                'algorithm': 'DBSCAN',
                'eps': 0.3,
                'min_samples': 2,
                'metric': 'cosine'
            },
            'table_heavy': {
                'algorithm': 'AgglomerativeClustering',
                'n_clusters': None,
                'distance_threshold': 0.7,
                'linkage': 'average'
            },
            'technical': {
                'algorithm': 'DBSCAN',
                'eps': 0.4,
                'min_samples': 3,
                'metric': 'cosine'
            },
            'narrative': {
                'algorithm': 'AgglomerativeClustering',
                'n_clusters': None,
                'distance_threshold': 0.6,
                'linkage': 'ward'
            },
            'mixed': {
                'algorithm': 'DBSCAN',
                'eps': 0.45,
                'min_samples': 2,
                'metric': 'cosine'
            }
        },
        'caching': {
            'max_cache_size': 1000,
            'cache_ttl_seconds': 3600,
            'enable_persistent_cache': False
        },
        'similarity': {
            'min_similarity_threshold': 0.3,
            'top_similar_chunks': 3
        }
    }
    settings = None
    
    # Basic logger fallback
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("semantic_clustering")

# =============================================================================
# TYPE DEFINITIONS & PROTOCOLS
# =============================================================================

@runtime_checkable
class ChunkProtocol(Protocol):
    """Protocol definition for chunk objects"""
    id: str
    text: str
    heading_text: str
    full_path: str
    importance_score: float
    start_position: int
    metadata: Dict[str, Any]

# ContentType now imported from content_types.py

@dataclass
class EmbeddingCacheEntry:
    """Cache entry for embeddings with TTL support.

    Typing uses Any for broader compatibility when numpy isn't installed.
    """
    embeddings: Any
    timestamp: float
    text_hash: str

    def is_expired(self, ttl_seconds: int) -> bool:
        return (time.time() - self.timestamp) > ttl_seconds

@dataclass
class SemanticClusterResult:
    """Result object for semantic clustering operations"""
    enhanced_chunks: List[ChunkProtocol]
    cluster_labels: Any
    similarity_matrix: Any
    embeddings: Any
    processing_time: float
    metadata: Dict[str, Any]

@dataclass
class ClusteringReport:
    """Comprehensive clustering analysis report"""
    summary: Dict[str, Any]
    cluster_details: Dict[int, Dict[str, Any]]
    recommendations: List[str]
    quality_metrics: Dict[str, float]
    
# =============================================================================
# SEMANTIC CLUSTERING ENHANCEMENT
# =============================================================================

class SemanticClusteringEnhancer:
    """Erweitert hierarchisches Chunking um semantische Clustering-Fähigkeiten
    
    Diese Klasse bietet erweiterte semantische Analyse für Dokument-Chunks
    mit konfigurierbaren Clustering-Algorithmen und intelligentem Caching.
    
    Attributes:
        embedding_model: SentenceTransformer Modell für Embeddings
        clustering_strategies: Konfiguration für verschiedene Content-Typen
        embedding_cache: LRU Cache für Embeddings mit TTL
        config: Konfiguration aus SEMANTIC_CONFIG
    """
    
    def __init__(self, model_name: Optional[str] = None, max_cache_size: int = None, clustering_method: Optional[str] = None) -> None:
        """Initialisiert den SemanticClusteringEnhancer
        
        Args:
            model_name: Name des SentenceTransformer Modells (optional)
            max_cache_size: Maximale Anzahl Cache-Einträge (optional)
            clustering_method: Clustering-Methode (optional, default: 'kmeans')
            
        Raises:
            ImportError: Wenn erforderliche Dependencies nicht verfügbar sind
            RuntimeError: Wenn kein SentenceTransformer Modell geladen werden kann
        """
        if not SEMANTIC_DEPS_AVAILABLE:
            raise ImportError(
                "Semantic chunking dependencies not available. "
                "Install with: pip install sentence-transformers scikit-learn"
            )
            
        self.logger = logger
        # Nutze zentrale Konfiguration falls verfügbar
        if settings and hasattr(settings, 'semantic'):
            self.config = {
                'models': settings.semantic.models,
                'batch_processing': settings.semantic.batch_processing,
                'clustering': settings.semantic.clustering,
                'caching': settings.semantic.caching,
                'similarity': settings.semantic.similarity
            }
        else:
            self.config = SEMANTIC_CONFIG
        
        # Cache-Konfiguration
        cache_size = max_cache_size or self.config['caching']['max_cache_size']
        self.embedding_cache: Dict[str, EmbeddingCacheEntry] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Modell-Konfiguration aus config
        self.models = self.config['models']
        
        # Standard-Modell laden
        model_to_load = model_name or SENTENCE_TRANSFORMER_MODEL or self.models['multilingual']
        self.embedding_model = self._load_embedding_model(model_to_load)
        
        # Speichere model_name als Instanzattribut
        self.model_name = model_to_load
        
        # Speichere clustering_method
        self.clustering_method = clustering_method or 'kmeans'
        
        # Clustering-Strategien aus Config
        self.clustering_strategies = self.config['clustering']
        
        # Performance-Metriken
        self._total_embeddings_generated = 0
        self._total_processing_time = 0.0
        
        self.logger.info(
            "SemanticClusteringEnhancer initialized",
            model=model_to_load,
            cache_size=cache_size,
            clustering_method=self.clustering_method,
            strategies=list(self.clustering_strategies.keys()),
            config_source="central" if settings and hasattr(settings, 'semantic') else "fallback"
        )
    
    def _load_embedding_model(self, model_name: str) -> Optional[Any]:
        """Lädt SentenceTransformer Modell mit Fallback-Strategie
        
        Args:
            model_name: Name des zu ladenden Modells
            
        Returns:
            Geladenes SentenceTransformer Modell oder None bei Fehler
            
        Raises:
            RuntimeError: Wenn kein Modell geladen werden kann
        """
        fallback_models = [
            model_name,
            self.models['multilingual'],
            self.models['fast_embedding']
        ]
        
        for model in fallback_models:
            try:
                embedding_model = SentenceTransformer(model)
                self.logger.info("SentenceTransformer model loaded successfully", model=model)
                return embedding_model
            except Exception as e:
                self.logger.warning(f"Failed to load model {model}: {e}")
                continue
        
        self.logger.error("Failed to load any SentenceTransformer model")
        raise RuntimeError("No SentenceTransformer model could be loaded")
    
    @lru_cache(maxsize=128)
    def _generate_text_hash(self, text: str) -> str:
        """Generiert Hash für Text-Caching
        
        Args:
            text: Text für Hash-Generierung
            
        Returns:
            SHA-256 Hash des Textes
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def _get_cached_embeddings(self, text_hash: str) -> Optional[Any]:
        """Holt Embeddings aus Cache falls verfügbar und nicht abgelaufen
        
        Args:
            text_hash: Hash des Textes
            
        Returns:
            Cached Embeddings oder None
        """
        if text_hash in self.embedding_cache:
            entry = self.embedding_cache[text_hash]
            ttl = self.config['caching']['cache_ttl_seconds']
            
            if not entry.is_expired(ttl):
                self._cache_hits += 1
                self.logger.debug("Using cached embeddings", text_hash=text_hash)
                return entry.embeddings
            else:
                # Abgelaufenen Eintrag entfernen
                del self.embedding_cache[text_hash]
                self.logger.debug("Cache entry expired", text_hash=text_hash)
        
        self._cache_misses += 1
        return None
    
    def _cache_embeddings(self, text_hash: str, embeddings: Any) -> None:
        """Speichert Embeddings im Cache
        
        Args:
            text_hash: Hash des Textes
            embeddings: Zu speichernde Embeddings
        """
        # Cache-Größe begrenzen (einfache LRU-Implementierung)
        max_size = self.config['caching']['max_cache_size']
        if len(self.embedding_cache) >= max_size:
            # Ältesten Eintrag entfernen
            oldest_key = min(self.embedding_cache.keys(), 
                           key=lambda k: self.embedding_cache[k].timestamp)
            del self.embedding_cache[oldest_key]
            self.logger.debug("Cache entry evicted", evicted_key=oldest_key)
        
        entry = EmbeddingCacheEntry(
            embeddings=embeddings,
            timestamp=time.time(),
            text_hash=text_hash
        )
        self.embedding_cache[text_hash] = entry
        self.logger.debug("Embeddings cached", text_hash=text_hash)
        
    def enhance_chunks_with_semantic_clustering(
        self, 
        chunks: List[ChunkProtocol], 
        content_type: Optional[ContentType] = None,
        use_hierarchical_context: bool = True,
        batch_size: Optional[int] = None
    ) -> SemanticClusterResult:
        """Hauptfunktion: Erweitert Chunks um semantische Clustering-Information
        
        Führt eine vollständige semantische Analyse der bereitgestellten Chunks durch,
        einschließlich Embedding-Generierung, Clustering und Ähnlichkeitsberechnung.
        
        Args:
            chunks: Liste der zu analysierenden Chunks
            content_type: Content-Typ für adaptive Clustering-Parameter
            use_hierarchical_context: Ob hierarchischer Kontext verwendet werden soll
            batch_size: Batch-Größe für Embedding-Generierung (optional)
            
        Returns:
            SemanticClusterResult mit erweiterten Chunks und Analysedaten
            
        Raises:
            ValueError: Wenn keine Chunks bereitgestellt werden
            RuntimeError: Wenn Embedding-Modell nicht verfügbar ist
        """
        start_time = time.time()
        
        if not chunks:
            raise ValueError("No chunks provided for semantic clustering")
            
        if not self.embedding_model:
            raise RuntimeError("No embedding model available")
        
        self.logger.info(
            "Starting semantic clustering",
            chunk_count=len(chunks),
            content_type=content_type.value if content_type else "auto",
            use_context=use_hierarchical_context
        )
        
        # 1. Generiere Embeddings für alle Chunks (mit Batch-Verarbeitung)
        embeddings = self._generate_chunk_embeddings_batched(
            chunks, 
            use_hierarchical_context,
            batch_size
        )
        
        # 2. Führe semantisches Clustering durch
        cluster_labels = self._perform_semantic_clustering(
            embeddings, 
            content_type or self._detect_content_type(chunks)
        )
        
        # 3. Berechne semantische Ähnlichkeitsmetriken
        similarity_matrix = self._calculate_semantic_similarities(embeddings)
        
        # 4. Erweitere Chunks mit Clustering-Informationen
        enhanced_chunks = self._enrich_chunks_with_semantic_data(
            chunks, 
            cluster_labels, 
            embeddings,
            similarity_matrix
        )
        
        # 5. Optimiere Chunks basierend auf semantischen Clustern
        optimized_chunks = self._optimize_chunks_by_semantic_clusters(enhanced_chunks)
        
        processing_time = time.time() - start_time
        self._total_processing_time += processing_time
        
        # Erstelle Ergebnis-Objekt
        result = SemanticClusterResult(
            enhanced_chunks=optimized_chunks,
            cluster_labels=cluster_labels,
            similarity_matrix=similarity_matrix,
            embeddings=embeddings,
            processing_time=processing_time,
            metadata={
                "content_type": content_type.value if content_type else "auto",
                "use_hierarchical_context": use_hierarchical_context,
                "batch_size_used": batch_size or self._get_optimal_batch_size(len(chunks)),
                "unique_clusters": len(set(cluster_labels)),
                "noise_chunks": sum(1 for label in cluster_labels if label == -1),
                "cache_hit_rate": self._get_cache_hit_rate()
            }
        )
        
        self.logger.info(
            "Semantic clustering complete",
            total_chunks=len(optimized_chunks),
            unique_clusters=len(set(cluster_labels)),
            noise_chunks=sum(1 for label in cluster_labels if label == -1),
            processing_time=f"{processing_time:.2f}s",
            cache_hit_rate=f"{self._get_cache_hit_rate():.1%}"
        )
        
        return result
    
    def _detect_content_type(self, chunks: List[ChunkProtocol]) -> ContentType:
        """Automatische Erkennung des Content-Typs basierend auf Chunk-Eigenschaften
        
        Args:
            chunks: Liste der Chunks zur Analyse
            
        Returns:
            Erkannter ContentType
        """
        if not chunks:
            return ContentType.MIXED
        
        # Analyse der Chunk-Eigenschaften
        total_text = " ".join(chunk.text for chunk in chunks[:5])  # Sample der ersten 5 Chunks
        
        # Rechtliche Indikatoren
        legal_indicators = ['§', 'artikel', 'gesetz', 'verordnung', 'paragraph', 'bestimmung']
        legal_score = sum(1 for indicator in legal_indicators if indicator.lower() in total_text.lower())
        
        # Technische Indikatoren
        tech_indicators = ['api', 'system', 'code', 'methode', 'funktion', 'parameter']
        tech_score = sum(1 for indicator in tech_indicators if indicator.lower() in total_text.lower())
        
        # Tabellen-Indikatoren (basierend auf Struktur)
        table_score = sum(1 for chunk in chunks[:10] if '|' in chunk.text or '\t' in chunk.text)
        
        # Narrative Indikatoren
        narrative_indicators = ['geschichte', 'erzählung', 'beispiel', 'fall', 'situation']
        narrative_score = sum(1 for indicator in narrative_indicators if indicator.lower() in total_text.lower())
        
        # Entscheidung basierend auf Scores
        scores = {
            ContentType.LEGAL_TEXT: legal_score,
            ContentType.TECHNICAL: tech_score,
            ContentType.TABLE_HEAVY: table_score,
            ContentType.NARRATIVE: narrative_score
        }
        
        detected_type = max(scores, key=scores.get)
        
        # Fallback zu MIXED wenn keine klare Tendenz
        if scores[detected_type] == 0:
            detected_type = ContentType.MIXED
        
        self.logger.debug(
            "Content type detected",
            detected_type=detected_type.value,
            scores=scores
        )
        
        return detected_type
    
    def _get_optimal_batch_size(self, chunk_count: int) -> int:
        """Bestimmt optimale Batch-Größe basierend auf Anzahl der Chunks
        
        Args:
            chunk_count: Anzahl der zu verarbeitenden Chunks
            
        Returns:
            Optimale Batch-Größe
        """
        config = self.config['batch_processing']
        
        if chunk_count < 20:
            return config['small_batch_size']
        elif chunk_count < config['max_chunks_for_batching']:
            return config['default_batch_size']
        else:
            return config['large_batch_size']
    
    def _get_cache_hit_rate(self) -> float:
        """Berechnet aktuelle Cache-Hit-Rate
        
        Returns:
            Cache-Hit-Rate als Decimal (0.0 - 1.0)
        """
        total_requests = self._cache_hits + self._cache_misses
        if total_requests == 0:
            return 0.0
        return self._cache_hits / total_requests
    
    def _generate_chunk_embeddings_batched(
        self, 
        chunks: List[ChunkProtocol], 
        use_hierarchical_context: bool = True,
        batch_size: Optional[int] = None
    ) -> Any:
        """Generiert Embeddings für Chunks mit Batch-Verarbeitung und Caching
        
        Args:
            chunks: Liste der Chunks
            use_hierarchical_context: Ob hierarchischer Kontext verwendet werden soll
            batch_size: Batch-Größe (optional, wird automatisch bestimmt)
            
        Returns:
            NumPy Array mit Embeddings
            
        Raises:
            RuntimeError: Wenn Embedding-Generierung fehlschlägt
        """
        if not chunks:
            return np.array([])
        
        batch_size = batch_size or self._get_optimal_batch_size(len(chunks))
        
        # Prepare texts for embedding
        texts_for_embedding = []
        text_hashes = []
        
        for chunk in chunks:
            if use_hierarchical_context:
                # Kombiniere Chunk-Text mit hierarchischem Kontext
                context_text = f"Heading: {chunk.heading_text}. "
                
                # Füge Pfad-Information hinzu für besseren Kontext
                if hasattr(chunk, 'full_path') and chunk.full_path:
                    context_text += f"Section: {chunk.full_path}. "
                
                # Füge Chunk-Content hinzu
                full_text = context_text + chunk.text
            else:
                full_text = chunk.text
            
            texts_for_embedding.append(full_text)
            text_hashes.append(self._generate_text_hash(full_text))
        
        # Check cache for all texts
        embeddings_list = []
        texts_to_process = []
        indices_to_process = []
        
        for i, text_hash in enumerate(text_hashes):
            cached_embedding = self._get_cached_embeddings(text_hash)
            if cached_embedding is not None:
                embeddings_list.append((i, cached_embedding))
            else:
                texts_to_process.append(texts_for_embedding[i])
                indices_to_process.append(i)
        
        # Process uncached texts in batches
        if texts_to_process:
            try:
                self.logger.debug(
                    "Generating embeddings",
                    total_texts=len(texts_for_embedding),
                    cached=len(embeddings_list),
                    to_process=len(texts_to_process),
                    batch_size=batch_size
                )
                
                # Process in batches to avoid memory issues
                processed_embeddings = []
                for i in range(0, len(texts_to_process), batch_size):
                    batch_texts = texts_to_process[i:i + batch_size]
                    
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts,
                        batch_size=len(batch_texts),
                        show_progress_bar=len(texts_to_process) > 50,
                        convert_to_numpy=True,
                        normalize_embeddings=True  # Normalize for better similarity computation
                    )
                    
                    processed_embeddings.extend(batch_embeddings)
                
                # Cache new embeddings
                for i, embedding in enumerate(processed_embeddings):
                    original_index = indices_to_process[i]
                    text_hash = text_hashes[original_index]
                    self._cache_embeddings(text_hash, embedding)
                    embeddings_list.append((original_index, embedding))
                
                self._total_embeddings_generated += len(processed_embeddings)
                
            except Exception as e:
                self.logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
                raise RuntimeError(f"Embedding generation failed: {e}") from e
        
        # Sort embeddings by original order and combine
        embeddings_list.sort(key=lambda x: x[0])
        final_embeddings = np.array([embedding for _, embedding in embeddings_list])
        
        self.logger.debug(
            "Embeddings generated",
            shape=final_embeddings.shape,
            cache_hit_rate=f"{self._get_cache_hit_rate():.1%}"
        )
        
        return final_embeddings
    
    def _perform_semantic_clustering(
        self, 
    embeddings: Any, 
        content_type: Optional[ContentType] = None
    ) -> Any:
        """Führt semantisches Clustering basierend auf Content-Type durch
        
        Args:
            embeddings: NumPy Array mit Embeddings
            content_type: Content-Typ für Clustering-Strategie-Auswahl
            
        Returns:
            NumPy Array mit Cluster-Labels für jeden Chunk
            
        Raises:
            ValueError: Wenn Embeddings leer sind
            RuntimeError: Wenn Clustering fehlschlägt
        """
        if embeddings.size == 0:
            raise ValueError("No embeddings provided for clustering")
        
        # Bestimme Clustering-Strategie
        strategy_key = content_type.value if content_type else 'mixed'
        strategy = self.clustering_strategies.get(strategy_key, self.clustering_strategies['mixed'])
        
        self.logger.debug(
            "Performing semantic clustering",
            strategy=strategy,
            content_type=strategy_key,
            embedding_shape=embeddings.shape
        )
        
        try:
            if strategy['algorithm'] == 'DBSCAN':
                clustering = DBSCAN(
                    eps=strategy['eps'],
                    min_samples=strategy['min_samples'],
                    metric=strategy.get('metric', 'euclidean')
                )
                cluster_labels = clustering.fit_predict(embeddings)
                
            elif strategy['algorithm'] == 'AgglomerativeClustering':
                if strategy.get('n_clusters'):
                    clustering = AgglomerativeClustering(
                        n_clusters=strategy['n_clusters'],
                        linkage=strategy.get('linkage', 'ward')
                    )
                else:
                    clustering = AgglomerativeClustering(
                        distance_threshold=strategy['distance_threshold'],
                        n_clusters=None,
                        linkage=strategy.get('linkage', 'ward')
                    )
                cluster_labels = clustering.fit_predict(embeddings)
            
            else:
                # Fallback zu DBSCAN
                self.logger.warning("Unknown clustering algorithm, falling back to DBSCAN")
                clustering = DBSCAN(eps=0.5, min_samples=2)
                cluster_labels = clustering.fit_predict(embeddings)
            
            unique_clusters = len(set(cluster_labels))
            noise_points = sum(1 for label in cluster_labels if label == -1)
            
            self.logger.debug(
                "Clustering completed",
                unique_clusters=unique_clusters,
                noise_points=noise_points,
                total_points=len(cluster_labels)
            )
            
            return cluster_labels
            
        except Exception as e:
            self.logger.error(f"Clustering failed with strategy {strategy}: {e}", exc_info=True)
            # Fallback: Alle Chunks als separate Cluster
            return np.arange(len(embeddings))
    
    def _calculate_semantic_similarities(self, embeddings: Any) -> Any:
        """Berechnet Cosine-Ähnlichkeitsmatrix zwischen allen Chunk-Embeddings
        
        Args:
            embeddings: NumPy Array mit normalisierten Embeddings
            
        Returns:
            Symmetrische Ähnlichkeitsmatrix (n x n)
            
        Raises:
            ValueError: Wenn Embeddings leer sind
            RuntimeError: Wenn Ähnlichkeitsberechnung fehlschlägt
        """
        if embeddings.size == 0:
            raise ValueError("No embeddings provided for similarity calculation")
        
        try:
            # Berechne Cosine-Ähnlichkeit (optimiert für normalisierte Embeddings)
            similarity_matrix = cosine_similarity(embeddings)
            
            self.logger.debug(
                "Similarity matrix calculated",
                shape=similarity_matrix.shape,
                avg_similarity=float(np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]))
            )
            
            return similarity_matrix
            
        except Exception as e:
            self.logger.error(f"Failed to calculate similarities: {e}", exc_info=True)
            # Fallback: Identity matrix
            return np.eye(len(embeddings))
    
    def _enrich_chunks_with_semantic_data(
        self,
        chunks: List[ChunkProtocol],
    cluster_labels: Any,
    embeddings: Any,
    similarity_matrix: Any
    ) -> List[ChunkProtocol]:
        """Erweitert Chunks um semantische Clustering-Daten
        
        Args:
            chunks: Liste der ursprünglichen Chunks
            cluster_labels: Cluster-Zuordnungen für jeden Chunk
            embeddings: Embedding-Vektoren für jeden Chunk
            similarity_matrix: Ähnlichkeitsmatrix zwischen allen Chunks
            
        Returns:
            Liste der mit semantischen Daten erweiterten Chunks
            
        Raises:
            ValueError: Wenn Eingabedaten inkonsistent sind
        """
        if len(chunks) != len(cluster_labels) or len(chunks) != len(embeddings):
            raise ValueError("Inconsistent input data lengths for chunk enrichment")
        
        config = self.config['similarity']
        min_similarity = config['min_similarity_threshold']
        top_k = config['top_similar_chunks']
        
        for i, chunk in enumerate(chunks):
            cluster_id = int(cluster_labels[i])
            
            # Grundlegende Cluster-Information
            chunk.metadata['semantic_cluster'] = cluster_id
            chunk.metadata['is_noise'] = cluster_id == -1  # DBSCAN Noise Detection
            
            # Embedding-Information (optional, kann groß werden)
            if len(embeddings[i]) < 1000:  # Nur bei kleinen Embeddings speichern
                chunk.metadata['embedding_vector'] = embeddings[i].tolist()
            chunk.metadata['embedding_dimension'] = embeddings.shape[1]
            
            # Ähnlichkeitsmetriken zu anderen Chunks
            chunk_similarities = similarity_matrix[i]
            
            # Finde ähnlichste Chunks (Top K, exklusive sich selbst)
            similar_indices = np.argsort(chunk_similarities)[::-1][1:top_k + 1]  # Skip self
            similar_chunks = [
                {
                    'chunk_id': chunks[idx].id,
                    'similarity_score': float(chunk_similarities[idx]),
                    'heading': chunks[idx].heading_text[:50]  # Begrenzte Länge
                }
                                for idx in similar_indices
                if idx < len(chunks) and chunk_similarities[idx] > min_similarity
            ]
            
            chunk.metadata['most_similar_chunks'] = similar_chunks
            
            # Cluster-spezifische Metriken
            if cluster_id != -1:  # Nicht-Noise Chunks
                cluster_mask = cluster_labels == cluster_id
                
                if np.any(cluster_mask):
                    cluster_embeddings = embeddings[cluster_mask]
                    
                    # Zentroid des Clusters
                    cluster_centroid = np.mean(cluster_embeddings, axis=0)
                    
                    # Distanz zum Cluster-Zentrum
                    chunk_distance_to_centroid = cosine_similarity(
                        [embeddings[i]], 
                        [cluster_centroid]
                    )[0][0]
                    
                    chunk.metadata['cluster_centrality'] = float(chunk_distance_to_centroid)
                    chunk.metadata['cluster_size'] = int(np.sum(cluster_mask))
                else:
                    chunk.metadata['cluster_centrality'] = 0.0
                    chunk.metadata['cluster_size'] = 1
            else:
                chunk.metadata['cluster_centrality'] = 0.0
                chunk.metadata['cluster_size'] = 1
        
        self.logger.debug(
            "Chunks enriched with semantic data",
            total_chunks=len(chunks),
            noise_chunks=sum(1 for label in cluster_labels if label == -1)
        )
        
        return chunks
    
    def _optimize_chunks_by_semantic_clusters(
        self, 
        chunks: List['HierarchicalChunk']
    ) -> List['HierarchicalChunk']:
        """Optimiert Chunks basierend auf semantischen Clustern"""
        
        # Gruppiere Chunks nach semantischen Clustern
        cluster_groups = {}
        for chunk in chunks:
            cluster_id = chunk.metadata.get('semantic_cluster', -1)
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(chunk)
        
        optimized_chunks = []
        
        for cluster_id, cluster_chunks in cluster_groups.items():
            if cluster_id == -1:  # Noise chunks - keine Optimierung
                optimized_chunks.extend(cluster_chunks)
                continue
            
            # Sortiere Chunks im Cluster nach Position (Dokumentreihenfolge beibehalten)
            cluster_chunks.sort(key=lambda c: c.start_position)
            
            # Berechne Cluster-Kohäsion
            cluster_cohesion = self._calculate_cluster_cohesion(cluster_chunks)
            
            # Aktualisiere Importance Score basierend auf Cluster-Eigenschaften
            for chunk in cluster_chunks:
                # Chunks in kohärenten Clustern sind wichtiger
                cohesion_bonus = cluster_cohesion * 0.2
                chunk.importance_score += cohesion_bonus
                
                # Zentrale Chunks im Cluster sind wichtiger
                centrality_bonus = chunk.metadata.get('cluster_centrality', 0.0) * 0.1
                chunk.importance_score += centrality_bonus
                
                # Füge Cluster-Informationen zur Metadata hinzu
                chunk.metadata['cluster_cohesion'] = float(cluster_cohesion)
                chunk.metadata['cluster_id'] = cluster_id
            
            optimized_chunks.extend(cluster_chunks)
        
        # Sortiere finale Chunks nach ursprünglicher Dokumentreihenfolge
        optimized_chunks.sort(key=lambda c: c.start_position)
        
        return optimized_chunks
    
    def _calculate_cluster_cohesion(self, cluster_chunks: List['HierarchicalChunk']) -> float:
        """Berechnet die semantische Kohäsion eines Clusters"""
        
        if len(cluster_chunks) < 2:
            return 1.0
        
        similarities = []
        for i, chunk1 in enumerate(cluster_chunks):
            similar_chunks = chunk1.metadata.get('most_similar_chunks', [])
            for similar_info in similar_chunks:
                if any(c.id == similar_info['chunk_id'] for c in cluster_chunks):
                    similarities.append(similar_info['similarity_score'])
        
        return np.mean(similarities) if similarities else 0.5
    
    def generate_semantic_clustering_report(
        self, 
        chunks: List[ChunkProtocol]
    ) -> ClusteringReport:
        """Generiert detaillierten Bericht über semantisches Clustering
        
        Args:
            chunks: Liste der analysierten Chunks mit semantischen Metadaten
            
        Returns:
            ClusteringReport mit umfassender Analyse der Clustering-Ergebnisse
            
        Raises:
            ValueError: Wenn keine Chunks bereitgestellt werden
        """
        if not chunks:
            raise ValueError("No chunks provided for report generation")
        
        cluster_stats = {}
        total_chunks = len(chunks)
        noise_chunks = 0
        total_similarity = 0.0
        similarity_count = 0
        
        # Sammle Cluster-Statistiken
        for chunk in chunks:
            cluster_id = chunk.metadata.get('semantic_cluster', -1)
            
            if cluster_id == -1:
                noise_chunks += 1
                continue
            
            if cluster_id not in cluster_stats:
                cluster_stats[cluster_id] = {
                    'chunk_count': 0,
                    'avg_importance': 0.0,
                    'avg_centrality': 0.0,
                    'headings': [],
                    'cohesion': 0.0,
                    'total_text_length': 0
                }
            
            stats = cluster_stats[cluster_id]
            stats['chunk_count'] += 1
            stats['avg_importance'] += chunk.importance_score
            stats['avg_centrality'] += chunk.metadata.get('cluster_centrality', 0.0)
            stats['headings'].append(chunk.heading_text[:30])  # Begrenzte Länge
            stats['cohesion'] = chunk.metadata.get('cluster_cohesion', 0.0)
            stats['total_text_length'] += len(chunk.text)
            
            # Sammle Ähnlichkeits-Scores für Qualitätsmetriken
            similar_chunks = chunk.metadata.get('most_similar_chunks', [])
            for similar in similar_chunks:
                total_similarity += similar['similarity_score']
                similarity_count += 1
        
        # Berechne Durchschnittswerte
        for cluster_id, stats in cluster_stats.items():
            if stats['chunk_count'] > 0:
                stats['avg_importance'] /= stats['chunk_count']
                stats['avg_centrality'] /= stats['chunk_count']
                stats['avg_text_length'] = stats['total_text_length'] / stats['chunk_count']
        
        # Qualitätsmetriken berechnen
        clustering_effectiveness = (total_chunks - noise_chunks) / total_chunks if total_chunks > 0 else 0.0
        avg_similarity = total_similarity / similarity_count if similarity_count > 0 else 0.0
        
        # Cluster-Balance (wie gleichmäßig sind Cluster-Größen?)
        cluster_sizes = [stats['chunk_count'] for stats in cluster_stats.values()]
        cluster_balance = 1.0 - (np.std(cluster_sizes) / np.mean(cluster_sizes)) if cluster_sizes and np.mean(cluster_sizes) > 0 else 0.0
        
        quality_metrics = {
            'clustering_effectiveness': clustering_effectiveness,
            'average_similarity': avg_similarity,
            'cluster_balance': max(0.0, cluster_balance),  # Negativ vermeiden
            'noise_ratio': noise_chunks / total_chunks if total_chunks > 0 else 0.0,
            'silhouette_score': self._calculate_silhouette_score_estimate(chunks) if len(chunks) > 1 else 0.0
        }
        
        summary = {
            'total_chunks': total_chunks,
            'unique_clusters': len(cluster_stats),
            'noise_chunks': noise_chunks,
            'clustering_effectiveness': clustering_effectiveness,
            'avg_chunks_per_cluster': np.mean(cluster_sizes) if cluster_sizes else 0.0,
            'largest_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'smallest_cluster_size': min(cluster_sizes) if cluster_sizes else 0
        }
        
        recommendations = self._generate_clustering_recommendations(
            cluster_stats, noise_chunks, total_chunks, quality_metrics
        )
        
        report = ClusteringReport(
            summary=summary,
            cluster_details=cluster_stats,
            recommendations=recommendations,
            quality_metrics=quality_metrics
        )
        
        self.logger.info(
            "Clustering report generated",
            total_chunks=total_chunks,
            unique_clusters=len(cluster_stats),
            effectiveness=f"{clustering_effectiveness:.1%}",
            avg_similarity=f"{avg_similarity:.3f}"
        )
        
        return report
    
    def _calculate_silhouette_score_estimate(self, chunks: List[ChunkProtocol]) -> float:
        """Schätzt Silhouette Score basierend auf Chunk-Metadaten
        
        Args:
            chunks: Liste der Chunks mit Clustering-Informationen
            
        Returns:
            Geschätzter Silhouette Score (0.0 - 1.0)
        """
        try:
            # Vereinfachte Silhouette-Schätzung basierend auf Cluster-Zentralität
            centralities = [chunk.metadata.get('cluster_centrality', 0.0) for chunk in chunks]
            noise_chunks = [chunk for chunk in chunks if chunk.metadata.get('is_noise', False)]
            
            # Penalty für Noise Chunks
            noise_penalty = len(noise_chunks) / len(chunks) if chunks else 0.0
            
            # Durchschnittliche Zentralität als Approximation
            avg_centrality = np.mean(centralities) if centralities else 0.0
            
            # Score zwischen 0 und 1, penalisiert durch Noise
            estimated_score = max(0.0, avg_centrality - noise_penalty)
            
            return float(estimated_score)
            
        except Exception as e:
            self.logger.warning(f"Failed to estimate silhouette score: {e}")
            return 0.0
    
    def _generate_clustering_recommendations(
        self, 
        cluster_stats: Dict, 
        noise_chunks: int, 
        total_chunks: int,
        quality_metrics: Dict[str, float]
    ) -> List[str]:
        """Generiert Empfehlungen basierend auf Clustering-Ergebnissen"""
        
        recommendations = []
        
        noise_ratio = noise_chunks / total_chunks
        if noise_ratio > 0.3:
            recommendations.append(
                f"⚠️ High noise ratio ({noise_ratio:.1%}). Consider adjusting clustering parameters."
            )
        
        if len(cluster_stats) == 1:
            recommendations.append(
                "🔍 Only one cluster found. Document might be very homogeneous or parameters too strict."
            )
        elif len(cluster_stats) > total_chunks * 0.8:
            recommendations.append(
                "📊 Too many small clusters. Consider relaxing clustering parameters."
            )
        
        # Finde beste und schlechteste Cluster
        if cluster_stats:
            best_cluster = max(cluster_stats.items(), key=lambda x: x[1]['cohesion'])
            worst_cluster = min(cluster_stats.items(), key=lambda x: x[1]['cohesion'])
            
            recommendations.append(
                f"🏆 Best cluster #{best_cluster[0]} with {best_cluster[1]['cohesion']:.2f} cohesion"
            )
            
            if worst_cluster[1]['cohesion'] < 0.3:
                recommendations.append(
                    f"⚡ Cluster #{worst_cluster[0]} has low cohesion ({worst_cluster[1]['cohesion']:.2f}). Consider review."
                )
        
        return recommendations

    def get_performance_stats(self) -> Dict[str, Any]:
        """Gibt Performance-Statistiken der Clustering-Engine zurück
        
        Returns:
            Dictionary mit Performance-Metriken
        """
        return {
            'cache_stats': {
                'hits': self._cache_hits,
                'misses': self._cache_misses,
                'hit_rate': self._get_cache_hit_rate(),
                'cache_size': len(self.embedding_cache)
            },
            'processing_stats': {
                'total_embeddings_generated': self._total_embeddings_generated,
                'total_processing_time': self._total_processing_time,
                'avg_processing_time': self._total_processing_time / max(1, self._total_embeddings_generated)
            },
            'model_info': {
                'embedding_model': str(self.embedding_model) if self.embedding_model else "None",
                'available_strategies': list(self.clustering_strategies.keys())
            },
            'config': {
                'cache_max_size': self.config['caching']['max_cache_size'],
                'batch_sizes': self.config['batch_processing'],
                'similarity_config': self.config['similarity']
            }
        }
    
    def clear_cache(self) -> None:
        """Löscht den Embedding-Cache"""
        cache_size_before = len(self.embedding_cache)
        self.embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        
        self.logger.info(
            "Cache cleared",
            entries_removed=cache_size_before
        )
    
    def cluster_texts(self, texts: List[str], n_clusters: int = None, clustering_method: Optional[str] = None) -> List[int]:
        """Clustert eine Liste von Texten und gibt Cluster-IDs zurück
        
        Args:
            texts: Liste der zu clusternden Texte
            n_clusters: Anzahl der gewünschten Cluster (optional)
            clustering_method: Clustering-Methode (optional, nutzt self.clustering_method als default)
            
        Returns:
            Liste von Cluster-IDs für jeden Text
            
        Raises:
            ValueError: Wenn keine Texte bereitgestellt werden
        """
        if not texts:
            return []
        
        # Generiere Embeddings für die Texte
        embeddings = self.embedding_model.encode(texts)
        
        clusters: List[int]
        try:
            method = clustering_method or getattr(self, 'clustering_method', 'kmeans')
            
            if method.lower() == 'kmeans':
                from sklearn.cluster import KMeans
                num_clusters = n_clusters if n_clusters is not None else min(len(texts), 5)
                labels = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(embeddings)
                clusters = [int(x) for x in labels]
            else:
                # Fallback: alle Texte in einen Cluster
                clusters = [0] * len(texts)
                
        except Exception as e:
            self.logger.warning(f"Clustering failed, using fallback: {e}")
            # Fallback ohne Dependencies - einfache Modulo-Verteilung
            n_clusters_fallback = n_clusters or 5
            clusters = [i % n_clusters_fallback for i in range(len(texts))]
        
        # Stelle sicher, dass die Länge stimmt
        if len(clusters) != len(texts):
            clusters = clusters[:len(texts)]
        
        return clusters
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Berechnet die semantische Ähnlichkeit zwischen zwei Texten
        
        Args:
            text1: Erster Text
            text2: Zweiter Text
            
        Returns:
            Ähnlichkeits-Score zwischen 0.0 und 1.0
        """
        if not text1 or not text2:
            return 0.0
        
        try:
            # Generiere Embeddings für beide Texte
            embeddings = self.embedding_model.encode([text1, text2])
            
            # Nehme ersten zwei Vektoren (Mock liefert evtl. mehr)
            if len(embeddings) >= 2:
                vec1, vec2 = embeddings[0], embeddings[1]
            elif len(embeddings) == 1:
                vec1 = vec2 = embeddings[0]
            else:
                return 0.0
            
            # Berechne Kosinus-Ähnlichkeit
            import math
            
            try:
                # Versuche torch first
                import torch
                if isinstance(vec1, torch.Tensor) and isinstance(vec2, torch.Tensor):
                    dot = float((vec1 * vec2).sum())
                    norm1 = float(torch.sqrt((vec1 * vec1).sum()))
                    norm2 = float(torch.sqrt((vec2 * vec2).sum()))
                else:
                    raise ImportError
            except ImportError:
                try:
                    # Versuche numpy
                    import numpy as np
                    vec1_np, vec2_np = np.array(vec1), np.array(vec2)
                    dot = float(np.dot(vec1_np, vec2_np))
                    norm1 = float(np.linalg.norm(vec1_np))
                    norm2 = float(np.linalg.norm(vec2_np))
                except Exception:
                    # Pure Python fallback
                    vec1_list, vec2_list = list(vec1), list(vec2)
                    dot = sum(a * b for a, b in zip(vec1_list, vec2_list))
                    norm1 = math.sqrt(sum(a * a for a in vec1_list))
                    norm2 = math.sqrt(sum(b * b for b in vec2_list))
            
            # Vermeide Division durch Null
            if norm1 == 0.0 or norm2 == 0.0:
                return 0.0
            
            similarity = dot / (norm1 * norm2)
            
            # Stelle sicher, dass der Wert zwischen 0 und 1 liegt
            similarity = max(0.0, min(1.0, similarity))
            
            return float(similarity)
            
        except Exception as e:
            self.logger.warning(f"Similarity calculation failed: {e}")
            return 0.0

# =============================================================================
# INTEGRATION IN HIERARCHICAL CHUNKER
# =============================================================================

# Diese Erweiterung würde in die bestehende HierarchicalChunker Klasse integriert:

def enhance_hierarchical_chunker():
    """
    Beispiel für Integration des semantischen Clusterings in den HierarchicalChunker
    """
    
    # Neue Methode für HierarchicalChunker:
    def process_document_with_semantic_enhancement(
        self, 
        text: str, 
        pdf_path: Optional[str] = None,
        preserve_hierarchy: bool = True,
        adaptive_sizing: bool = True,
        enable_semantic_clustering: bool = True
    ) -> Tuple[List['HierarchicalChunk'], Any, Dict[str, Any]]:
        """Erweiterte Dokumentverarbeitung mit semantischem Clustering"""
        
        # Ursprüngliche hierarchische Verarbeitung
        chunks, document_tree = self.process_document(
            text, pdf_path, preserve_hierarchy, adaptive_sizing
        )
        
        semantic_report = {}
        
        if enable_semantic_clustering and chunks:
            # Initialisiere semantisches Clustering
            semantic_enhancer = SemanticClusteringEnhancer()
            
            # Erweitere Chunks um semantische Informationen
            enhanced_chunks = semantic_enhancer.enhance_chunks_with_semantic_clustering(
                chunks=chunks,
                content_type=self.toc_parser.detect_content_type(text),
                use_hierarchical_context=True
            )
            
            # Generiere Clustering-Bericht
            semantic_report = semantic_enhancer.generate_semantic_clustering_report(
                enhanced_chunks
            )
            
            self.logger.info(
                "Semantic enhancement complete",
                clusters_found=semantic_report.get('summary', {}).get('unique_clusters', 0),
                effectiveness=semantic_report.get('summary', {}).get('clustering_effectiveness', 0)
            )
            
            return enhanced_chunks, document_tree, semantic_report
        
        return chunks, document_tree, semantic_report

# =============================================================================
# DEMO UND TESTING
# =============================================================================

def demo_semantic_clustering():
    """Demonstriert die semantische Clustering-Funktionalität"""
    
    print("🧠 SEMANTIC CLUSTERING ENHANCEMENT DEMO")
    print("=" * 50)
    
    # Beispiel-Chunks erstellen (vereinfacht für Demo)
    from dataclasses import dataclass
    from enum import Enum
    from .content_types import ContentType
## ContentType now centralized in content_types.py
    
    @dataclass
    class MockChunk:
        id: str
        text: str
        heading_text: str
        full_path: str
        content_type: ContentType
        importance_score: float
        start_position: int
        metadata: dict
    
    # Erstelle Test-Chunks
    test_chunks = [
        MockChunk(
            id="chunk_1",
            text="Die Berufsunfähigkeitsversicherung bietet Schutz bei Arbeitsunfähigkeit.",
            heading_text="Versicherungsschutz",
            full_path="BU-Versicherung > Versicherungsschutz",
            content_type=ContentType.LEGAL_TEXT,
            importance_score=1.0,
            start_position=0,
            metadata={}
        ),
        MockChunk(
            id="chunk_2",
            text="Bei Berufsunfähigkeit zahlt der Versicherer eine monatliche Rente.",
            heading_text="Leistungen",
            full_path="BU-Versicherung > Leistungen",
            content_type=ContentType.LEGAL_TEXT,
            importance_score=1.0,
            start_position=100,
            metadata={}
        ),
        MockChunk(
            id="chunk_3",
            text="Die Gesundheitsprüfung erfolgt vor Vertragsabschluss durch einen Arzt.",
            heading_text="Gesundheitsprüfung",
            full_path="BU-Versicherung > Gesundheitsprüfung",
            content_type=ContentType.LEGAL_TEXT,
            importance_score=0.8,
            start_position=200,
            metadata={}
        )
    ]
    
    # Initialisiere Semantic Enhancer
    enhancer = SemanticClusteringEnhancer()
    
    # Führe semantisches Clustering durch
    enhanced_chunks = enhancer.enhance_chunks_with_semantic_clustering(
        chunks=test_chunks,
        content_type=ContentType.LEGAL_TEXT,
        use_hierarchical_context=True
    )
    
    # Zeige Ergebnisse
    print(f"📊 CLUSTERING RESULTS:")
    for chunk in enhanced_chunks:
        print(f"\n   Chunk: {chunk.id}")
        print(f"   Cluster: {chunk.metadata.get('semantic_cluster', 'N/A')}")
        print(f"   Centrality: {chunk.metadata.get('cluster_centrality', 0.0):.3f}")
        print(f"   Similar chunks: {len(chunk.metadata.get('most_similar_chunks', []))}")
        print(f"   Importance: {chunk.importance_score:.3f}")
    
    # Generiere und zeige Bericht
    report = enhancer.generate_semantic_clustering_report(enhanced_chunks)
    print(f"\n📈 CLUSTERING REPORT:")
    print(f"   Total clusters: {report['summary']['unique_clusters']}")
    print(f"   Effectiveness: {report['summary']['clustering_effectiveness']:.1%}")
    print(f"   Noise chunks: {report['summary']['noise_chunks']}")
    
    for rec in report['recommendations']:
        print(f"   {rec}")
    
    return enhanced_chunks, report

if __name__ == "__main__":
    demo_semantic_clustering()
