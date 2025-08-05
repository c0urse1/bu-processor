#!/usr/bin/env python3
"""
ERWEITERTE SIMHASH SEMANTISCHE DEDUPLICATION
==========================================

Verbesserte SimHash-Implementierung zur Erkennung semantischer Duplikate
mit korrekter Bit-Vektor Akkumulation und N-Gram-basierter Analyse.

Features:
- Korrekte SimHash-Algorithmus Implementierung
- N-Gram-basierte semantische Erfassung
- Hamming-Distanz für Ähnlichkeitsvergleiche
- Integration in bestehende Deduplication-Pipeline
- Performance-optimiert mit Caching
"""

import mmh3
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import re
from dataclasses import dataclass
import hashlib

# Config import mit zentraler Konfiguration
try:
    from ..core.config import (
        settings, LOG_LEVEL, DEDUPLICATION_CONFIG, SIMILARITY_THRESHOLD
    )
    # Logger Setup aus config
    import structlog
    import logging
    
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
    logger = structlog.get_logger("simhash_semantic_deduplication")
    
except ImportError:
    # Fallback values
    LOG_LEVEL = "INFO"
    SIMILARITY_THRESHOLD = 0.85
    
    # Fallback Konfiguration falls zentrale Config nicht verfügbar
    DEDUPLICATION_CONFIG = {
        'simhash_bit_size': 64,
        'simhash_ngram_size': 3,
        'similarity_threshold': 0.85,
        'hamming_threshold': 8,
        'content_type_weights': {
            'legal_text': 1.5,
            'technical_spec': 1.3,
            'narrative': 1.0,
            'table': 0.8,
            'list': 0.7,
            'mixed': 1.0
        },
        'selection_weights': {
            'text_length_factor': 0.001,
            'heading_level_factor': 10.0,
            'token_count_factor': 0.1
        },
        'enable_caching': True,
        'cache_size_limit': 10000,
        'enable_semantic_features': True,
        'min_important_word_length': 6,
        'min_token_length': 3
    }
    settings = None
    
    # Basic logger fallback
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("simhash_semantic_deduplication")

# Legacy imports - müssen angepasst werden für die korrekte Pipeline-Struktur
try:
    from .chunker import HierarchicalChunk, ContentType
except ImportError:
    # Fallback für Demo
    from dataclasses import dataclass
    from enum import Enum
    
    class ContentType(Enum):
        LEGAL_TEXT = "legal_text"
        TECHNICAL_SPEC = "technical_spec"
        NARRATIVE = "narrative"
        TABLE = "table"
        LIST = "list"
        MIXED = "mixed"
    
    @dataclass
    class HierarchicalChunk:
        id: str
        text: str
        content_type: ContentType
        heading_level: Optional[int] = None
        token_count: int = 0

@dataclass
class SimHashResult:
    """Ergebnis der SimHash-Berechnung mit Metadaten"""
    chunk_id: str
    simhash_value: int
    bit_vector: np.ndarray
    token_count: int
    ngram_count: int
    semantic_features: Dict[str, float]

class SemanticSimHashGenerator:
    """Erweiterte SimHash-Implementierung für semantische Duplikatserkennung"""
    
    def __init__(self, bit_size: Optional[int] = None, ngram_size: Optional[int] = None):
        # Nutze zentrale Konfiguration falls verfügbar
        if settings and hasattr(settings, 'deduplication'):
            self.config = {
                'simhash_bit_size': settings.deduplication.simhash_bit_size,
                'simhash_ngram_size': settings.deduplication.simhash_ngram_size,
                'content_type_weights': settings.deduplication.content_type_weights,
                'enable_caching': settings.deduplication.enable_caching,
                'cache_size_limit': settings.deduplication.cache_size_limit,
                'enable_semantic_features': settings.deduplication.enable_semantic_features,
                'min_important_word_length': settings.deduplication.min_important_word_length,
                'min_token_length': settings.deduplication.min_token_length
            }
        else:
            self.config = DEDUPLICATION_CONFIG
        
        self.bit_size = bit_size or self.config['simhash_bit_size']
        self.ngram_size = ngram_size or self.config['simhash_ngram_size']
        
        # Preprocessing patterns für bessere semantische Erfassung
        self.text_patterns = {
            'legal_terms': re.compile(r'\b(versicherung|vertrag|haftung|anspruch|leistung)\w*\b', re.IGNORECASE),
            'numbers': re.compile(r'\b\d+(?:[.,]\d+)*\b'),
            'dates': re.compile(r'\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b'),
            'currency': re.compile(r'\b\d+(?:[.,]\d+)*\s*(?:€|EUR|euro)\b', re.IGNORECASE)
        }
        
        # Cache für performance
        if self.config['enable_caching']:
            self._cache = {}
            self._cache_size_limit = self.config['cache_size_limit']
        else:
            self._cache = None
        
        logger.info("semantic_simhash_generator_initialized", 
                   bit_size=self.bit_size, 
                   ngram_size=self.ngram_size,
                   config_source="central" if settings and hasattr(settings, 'deduplication') else "fallback")
    
    def generate_simhash(self, chunk: HierarchicalChunk) -> SimHashResult:
        """
        Generiert korrekten SimHash mit semantischer Analyse
        
        Args:
            chunk: HierarchicalChunk zum Verarbeiten
            
        Returns:
            SimHashResult mit SimHash-Wert und Metadaten
        """
        
        if self._cache is not None:
            cache_key = f"simhash_{chunk.id}_{hash(chunk.text)}"
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Schritt 1: Text preprocessing und Feature-Extraktion
        if self.config['enable_semantic_features']:
            semantic_features = self._extract_semantic_features(chunk.text)
        else:
            semantic_features = {}
        
        # Schritt 2: N-Gram Generierung
        ngrams = self._generate_ngrams(chunk.text)
        
        # Schritt 3: Bit-Vektor Akkumulation (KORRIGIERTE IMPLEMENTIERUNG)
        bit_vector = np.zeros(self.bit_size, dtype=np.float64)
        
        # Gewichtung basierend auf Content-Type
        content_weight = self._get_content_type_weight(chunk.content_type)
        
        # Verarbeite N-Grams mit korrekter Bit-Akkumulation
        for ngram in ngrams:
            # Hash das N-Gram
            ngram_hash = mmh3.hash(ngram, signed=False)
            
            # Gewichtung basierend auf semantischen Features
            weight = content_weight
            
            # Erhöhe Gewichtung für wichtige Terme
            if self._is_important_term(ngram, semantic_features):
                weight *= 2.0
            
            # Akkumuliere Bits KORREKT (nicht überschreiben!)
            for bit_pos in range(self.bit_size):
                if ngram_hash & (1 << bit_pos):
                    bit_vector[bit_pos] += weight
                else:
                    bit_vector[bit_pos] -= weight
        
        # Schritt 4: Finaler SimHash-Wert generieren
        simhash_value = 0
        for bit_pos in range(self.bit_size):
            if bit_vector[bit_pos] > 0:
                simhash_value |= (1 << bit_pos)
        
        # Schritt 5: Ergebnis erstellen
        result = SimHashResult(
            chunk_id=chunk.id,
            simhash_value=simhash_value,
            bit_vector=bit_vector,
            token_count=len(chunk.text.split()),
            ngram_count=len(ngrams),
            semantic_features=semantic_features
        )
        
        # Cache das Ergebnis
        if self._cache is not None:
            # Cache-Größe begrenzen
            if len(self._cache) >= self._cache_size_limit:
                # Entferne ältesten Eintrag (einfache FIFO-Strategie)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[cache_key] = result
        
        logger.debug("simhash_generated", 
                    chunk_id=chunk.id,
                    simhash_hex=hex(simhash_value),
                    ngram_count=len(ngrams),
                    semantic_score=semantic_features.get('semantic_density', 0.0))
        
        return result
    
    def _extract_semantic_features(self, text: str) -> Dict[str, float]:
        """Extrahiert semantische Features aus dem Text"""
        
        features = {}
        
        # Text-Statistiken
        words = text.lower().split()
        word_count = len(words)
        char_count = len(text)
        
        features['word_density'] = (word_count / char_count * 100) if char_count > 0 else 0.0
        features['avg_word_length'] = (sum(len(w) for w in words) / word_count) if word_count > 0 else 0.0
        
        # Semantische Pattern-Erkennung
        for pattern_name, pattern in self.text_patterns.items():
            matches = pattern.findall(text)
            features[f'{pattern_name}_count'] = len(matches)
            features[f'{pattern_name}_density'] = (len(matches) / word_count) if word_count > 0 else 0.0
        
        # Lexikalische Diversität
        unique_words = set(words)
        features['lexical_diversity'] = (len(unique_words) / word_count) if word_count > 0 else 0.0
        
        # Semantische Dichte (kombinierte Metrik)
        features['semantic_density'] = (
            features['lexical_diversity'] * 0.4 +
            features.get('legal_terms_density', 0) * 0.3 +
            features['word_density'] / 100 * 0.3
        )
        
        return features
    
    def _generate_ngrams(self, text: str) -> List[str]:
        """Generiert N-Grams mit intelligentem Preprocessing"""
        
        # Text normalisierung
        normalized_text = self._normalize_text(text)
        tokens = normalized_text.split()
        
        if len(tokens) < self.ngram_size:
            # Für sehr kurze Texte: Character-level N-Grams
            return [normalized_text[i:i+self.ngram_size] 
                   for i in range(len(normalized_text) - self.ngram_size + 1)]
        
        # Standard Token-level N-Grams
        ngrams = []
        for i in range(len(tokens) - self.ngram_size + 1):
            ngram = ' '.join(tokens[i:i+self.ngram_size])
            ngrams.append(ngram)
        
        # Füge auch wichtige 1-Grams und 2-Grams hinzu für bessere Semantik
        for token in tokens:
            if self._is_important_single_token(token):
                ngrams.append(token)
        
        # 2-Grams für wichtige Begriffskombinationen
        if self.ngram_size > 2:
            for i in range(len(tokens) - 1):
                bigram = f"{tokens[i]} {tokens[i+1]}"
                if self._is_important_bigram(bigram):
                    ngrams.append(bigram)
        
        return list(set(ngrams))  # Entferne Duplikate
    
    def _normalize_text(self, text: str) -> str:
        """Intelligente Text-Normalisierung für semantische Analyse"""
        
        # Basis-Normalisierung
        normalized = text.lower().strip()
        
        # Spezielle Zeichen behandeln
        normalized = re.sub(r'[^\w\s.,!?-]', ' ', normalized)
        
        # Mehrfache Leerzeichen entfernen
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Zahlen normalisieren (behalte semantische Bedeutung)
        normalized = re.sub(r'\b\d+([.,]\d+)*\b', '<NUM>', normalized)
        
        return normalized
    
    def _is_important_term(self, ngram: str, semantic_features: Dict[str, float]) -> bool:
        """Bestimmt ob ein N-Gram semantisch wichtig ist"""
        
        # Legal/Fachbegriffe sind wichtiger
        if any(pattern.search(ngram) for pattern in self.text_patterns.values()):
            return True
        
        # Längere, seltene Wörter sind wichtiger
        words = ngram.split()
        avg_length = sum(len(w) for w in words) / len(words)
        if avg_length > 6:  # Längere Wörter sind oft spezifischer
            return True
        
        return False
    
    def _is_important_single_token(self, token: str) -> bool:
        """Bestimmt ob ein einzelner Token wichtig ist"""
        
        # Sehr kurze oder sehr häufige Wörter ignorieren
        min_length = self.config['min_token_length']
        if len(token) < min_length:
            return False
        
        # Stopwords (vereinfacht für Deutsch)
        stopwords = {'der', 'die', 'das', 'und', 'oder', 'aber', 'mit', 'von', 'bei', 'zu', 'im', 'am', 'ist', 'sind', 'haben', 'wird', 'werden', 'kann', 'soll'}
        if token in stopwords:
            return False
        
        # Fachbegriffe sind wichtig
        if any(pattern.search(token) for pattern in self.text_patterns.values()):
            return True
        
        # Längere Wörter sind oft wichtiger
        min_important_length = self.config['min_important_word_length']
        return len(token) >= min_important_length
    
    def _is_important_bigram(self, bigram: str) -> bool:
        """Bestimmt ob ein Bigram semantisch wichtig ist"""
        
        # Wichtige Begriffskombinationen
        important_patterns = [
            r'versicherung\w* \w+',
            r'\w+ vertrag',
            r'monatlich\w* \w+',
            r'\w+ leistung',
            r'anspruch \w+'
        ]
        
        return any(re.search(pattern, bigram, re.IGNORECASE) for pattern in important_patterns)
    
    def _get_content_type_weight(self, content_type: ContentType) -> float:
        """Gewichtung basierend auf Content-Type"""
        
        weights = self.config['content_type_weights']
        
        # Mapping von ContentType enum zu config keys
        type_mapping = {
            ContentType.LEGAL_TEXT: 'legal_text',
            ContentType.TECHNICAL_SPEC: 'technical_spec',
            ContentType.NARRATIVE: 'narrative',
            ContentType.TABLE: 'table',
            ContentType.LIST: 'list',
            ContentType.MIXED: 'mixed'
        }
        
        config_key = type_mapping.get(content_type, 'mixed')
        return weights.get(config_key, 1.0)

class SemanticDuplicateDetector:
    """Erweiterte Duplikatserkennung mit semantischem SimHash"""
    
    def __init__(self, similarity_threshold: Optional[float] = None, hamming_threshold: Optional[int] = None):
        # Nutze zentrale Konfiguration falls verfügbar
        if settings and hasattr(settings, 'deduplication'):
            self.config = {
                'similarity_threshold': settings.deduplication.similarity_threshold,
                'hamming_threshold': settings.deduplication.hamming_threshold,
                'simhash_bit_size': settings.deduplication.simhash_bit_size,
                'simhash_ngram_size': settings.deduplication.simhash_ngram_size
            }
        else:
            self.config = DEDUPLICATION_CONFIG
        
        self.similarity_threshold = similarity_threshold or self.config['similarity_threshold']
        self.hamming_threshold = hamming_threshold or self.config['hamming_threshold']
        
        self.simhash_generator = SemanticSimHashGenerator(
            bit_size=self.config.get('simhash_bit_size', 64),
            ngram_size=self.config.get('simhash_ngram_size', 3)
        )
        
        # Statistiken
        self._stats = {
            'chunks_processed': 0,
            'duplicates_found': 0,
            'near_duplicates_found': 0,
            'semantic_clusters': 0
        }
        
        logger.info("semantic_duplicate_detector_initialized",
                   similarity_threshold=self.similarity_threshold,
                   hamming_threshold=self.hamming_threshold,
                   config_source="central" if settings and hasattr(settings, 'deduplication') else "fallback")
    
    def find_semantic_duplicates(self, chunks: List[HierarchicalChunk]) -> Dict[str, List[str]]:
        """
        Findet semantische Duplikate mit erweitertem SimHash
        
        Args:
            chunks: Liste der zu verarbeitenden Chunks
            
        Returns:
            Dictionary mit Cluster-ID als Key und Liste der Chunk-IDs als Value
        """
        
        if not chunks:
            return {}
        
        logger.info("semantic_duplicate_detection_started", chunk_count=len(chunks))
        
        # Schritt 1: SimHash für alle Chunks generieren
        simhash_results = {}
        for chunk in chunks:
            try:
                result = self.simhash_generator.generate_simhash(chunk)
                simhash_results[chunk.id] = result
            except Exception as e:
                logger.warning("simhash_generation_failed", 
                             chunk_id=chunk.id, 
                             error=str(e))
                continue
        
        self._stats['chunks_processed'] = len(simhash_results)
        
        # Schritt 2: Duplikate durch Hamming-Distanz finden
        duplicate_clusters = self._find_hamming_clusters(simhash_results)
        
        # Schritt 3: Semantische Verfeinerung
        refined_clusters = self._refine_clusters_semantically(duplicate_clusters, simhash_results)
        
        # Statistiken aktualisieren
        self._stats['duplicates_found'] = sum(len(cluster) - 1 for cluster in refined_clusters.values() if len(cluster) > 1)
        self._stats['semantic_clusters'] = len(refined_clusters)
        
        logger.info("semantic_duplicate_detection_completed",
                   chunks_processed=self._stats['chunks_processed'],
                   clusters_found=len(refined_clusters),
                   duplicates_found=self._stats['duplicates_found'])
        
        return refined_clusters
    
    def _find_hamming_clusters(self, simhash_results: Dict[str, SimHashResult]) -> Dict[str, List[str]]:
        """Findet Cluster basierend auf Hamming-Distanz"""
        
        clusters = {}
        processed = set()
        cluster_id = 0
        
        chunk_ids = list(simhash_results.keys())
        
        for i, chunk_id1 in enumerate(chunk_ids):
            if chunk_id1 in processed:
                continue
            
            current_cluster = [chunk_id1]
            hash1 = simhash_results[chunk_id1].simhash_value
            
            # Finde ähnliche Chunks
            for j, chunk_id2 in enumerate(chunk_ids[i+1:], i+1):
                if chunk_id2 in processed:
                    continue
                
                hash2 = simhash_results[chunk_id2].simhash_value
                hamming_distance = self._calculate_hamming_distance(hash1, hash2)
                
                if hamming_distance <= self.hamming_threshold:
                    current_cluster.append(chunk_id2)
            
            # Nur Cluster mit mehreren Chunks speichern
            if len(current_cluster) > 1:
                clusters[f"semantic_cluster_{cluster_id}"] = current_cluster
                processed.update(current_cluster)
                cluster_id += 1
        
        return clusters
    
    def _calculate_hamming_distance(self, hash1: int, hash2: int) -> int:
        """Berechnet Hamming-Distanz zwischen zwei SimHash-Werten"""
        return bin(hash1 ^ hash2).count('1')
    
    def _refine_clusters_semantically(
        self, 
        clusters: Dict[str, List[str]], 
        simhash_results: Dict[str, SimHashResult]
    ) -> Dict[str, List[str]]:
        """Verfeinert Cluster mit semantischer Analyse"""
        
        refined_clusters = {}
        
        for cluster_id, chunk_ids in clusters.items():
            if len(chunk_ids) < 2:
                continue
            
            # Berechne semantische Ähnlichkeiten
            semantic_similarities = self._calculate_semantic_similarities(chunk_ids, simhash_results)
            
            # Filtere Cluster basierend auf semantischer Ähnlichkeit
            refined_cluster = self._filter_by_semantic_similarity(chunk_ids, semantic_similarities)
            
            if len(refined_cluster) > 1:
                refined_clusters[cluster_id] = refined_cluster
        
        return refined_clusters
    
    def _calculate_semantic_similarities(
        self, 
        chunk_ids: List[str], 
        simhash_results: Dict[str, SimHashResult]
    ) -> Dict[Tuple[str, str], float]:
        """Berechnet semantische Ähnlichkeiten zwischen Chunks"""
        
        similarities = {}
        
        for i, chunk_id1 in enumerate(chunk_ids):
            for j, chunk_id2 in enumerate(chunk_ids[i+1:], i+1):
                # Hamming-basierte Ähnlichkeit
                hash1 = simhash_results[chunk_id1].simhash_value
                hash2 = simhash_results[chunk_id2].simhash_value
                hamming_dist = self._calculate_hamming_distance(hash1, hash2)
                hamming_similarity = 1.0 - (hamming_dist / 64.0)
                
                # Feature-basierte Ähnlichkeit
                features1 = simhash_results[chunk_id1].semantic_features
                features2 = simhash_results[chunk_id2].semantic_features
                feature_similarity = self._calculate_feature_similarity(features1, features2)
                
                # Kombinierte Ähnlichkeit
                combined_similarity = (hamming_similarity * 0.7 + feature_similarity * 0.3)
                
                similarities[(chunk_id1, chunk_id2)] = combined_similarity
        
        return similarities
    
    def _calculate_feature_similarity(
        self, 
        features1: Dict[str, float], 
        features2: Dict[str, float]
    ) -> float:
        """Berechnet Ähnlichkeit zwischen semantischen Features"""
        
        # Gemeinsame Features finden
        common_features = set(features1.keys()) & set(features2.keys())
        
        if not common_features:
            return 0.0
        
        # Cosine Similarity für numerische Features
        vec1 = np.array([features1.get(f, 0.0) for f in common_features])
        vec2 = np.array([features2.get(f, 0.0) for f in common_features])
        
        # Vermeiden Division durch Null
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        return max(0.0, cosine_sim)  # Negative Werte auf 0 setzen
    
    def _filter_by_semantic_similarity(
        self, 
        chunk_ids: List[str], 
        similarities: Dict[Tuple[str, str], float]
    ) -> List[str]:
        """Filtert Cluster basierend auf semantischer Ähnlichkeit"""
        
        # Behalte nur Chunks, die mindestens die Ähnlichkeitsschwelle erreichen
        valid_chunks = set()
        
        for (chunk_id1, chunk_id2), similarity in similarities.items():
            if similarity >= self.similarity_threshold:
                valid_chunks.add(chunk_id1)
                valid_chunks.add(chunk_id2)
        
        # Rückgabe als sortierte Liste
        return sorted(list(valid_chunks))
    
    def get_stats(self) -> Dict[str, int]:
        """Gibt Statistiken der Duplikatserkennung zurück"""
        return self._stats.copy()

# Integration in bestehende Deduplication-Pipeline
class EnhancedDeduplicationEngine:
    """Erweiterte Deduplication-Engine mit semantischem SimHash"""
    
    def __init__(self, similarity_threshold: Optional[float] = None):
        # Nutze konfigurierte Schwellenwerte aus zentraler Config
        if settings and hasattr(settings, 'deduplication'):
            self.similarity_threshold = similarity_threshold or settings.deduplication.similarity_threshold
            self.selection_weights = settings.deduplication.selection_weights
        else:
            self.similarity_threshold = similarity_threshold or DEDUPLICATION_CONFIG['similarity_threshold']
            self.selection_weights = DEDUPLICATION_CONFIG['selection_weights']
        
        # Semantischer Duplikat-Detektor
        self.semantic_detector = SemanticDuplicateDetector(
            similarity_threshold=self.similarity_threshold,
            hamming_threshold=DEDUPLICATION_CONFIG.get('hamming_threshold', 8)
        )
        
        logger.info("enhanced_deduplication_engine_initialized",
                   similarity_threshold=self.similarity_threshold,
                   config_source="central" if settings and hasattr(settings, 'deduplication') else "fallback")
    
    def deduplicate_chunks_semantically(self, chunks: List[HierarchicalChunk]) -> Tuple[List[HierarchicalChunk], Dict[str, any]]:
        """
        Hauptmethode für semantische Deduplication
        
        Args:
            chunks: Liste der zu deduplizierenden Chunks
            
        Returns:
            Tuple aus (deduplizierte_chunks, report)
        """
        
        if not chunks:
            return chunks, {'error': 'No chunks provided'}
        
        logger.info("semantic_deduplication_started", 
                   original_chunks=len(chunks))
        
        # Finde semantische Duplikate
        duplicate_clusters = self.semantic_detector.find_semantic_duplicates(chunks)
        
        # Erstelle Chunk-Lookup
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        chunks_to_remove = set()
        
        # Verarbeite jeden Cluster
        for cluster_id, chunk_ids in duplicate_clusters.items():
            if len(chunk_ids) < 2:
                continue
            
            # Wähle den besten Chunk als Repräsentant
            representative = self._select_best_representative(
                [chunks_by_id[cid] for cid in chunk_ids if cid in chunks_by_id]
            )
            
            # Markiere andere Chunks zur Entfernung
            for chunk_id in chunk_ids:
                if chunk_id != representative.id:
                    chunks_to_remove.add(chunk_id)
            
            logger.debug("cluster_processed",
                        cluster_id=cluster_id,
                        chunk_count=len(chunk_ids),
                        representative=representative.id)
        
        # Erstelle deduplizierte Liste
        deduplicated_chunks = [
            chunk for chunk in chunks 
            if chunk.id not in chunks_to_remove
        ]
        
        # Erstelle Report
        deduplication_rate = len(chunks_to_remove) / len(chunks) if len(chunks) > 0 else 0.0
        report = {
            'original_chunks': len(chunks),
            'deduplicated_chunks': len(deduplicated_chunks),
            'removed_chunks': len(chunks_to_remove),
            'deduplication_rate': deduplication_rate,
            'clusters_found': len(duplicate_clusters),
            'detector_stats': self.semantic_detector.get_stats()
        }
        
        logger.info("semantic_deduplication_completed",
                   **report)
        
        return deduplicated_chunks, report
    
    def _select_best_representative(self, chunks: List[HierarchicalChunk]) -> HierarchicalChunk:
        """Wählt den besten Chunk als Repräsentant für einen Cluster"""
        
        if len(chunks) == 1:
            return chunks[0]
        
        # Bewertungskriterien aus zentraler Konfiguration
        weights = self.selection_weights
        
        # Bewertungskriterien
        def score_chunk(chunk):
            score = 0.0
            
            # Längere Chunks sind oft besser
            score += len(chunk.text) * weights['text_length_factor']
            
            # Höhere hierarchische Ebenen sind wichtiger
            if hasattr(chunk, 'heading_level') and chunk.heading_level:
                score += (7 - chunk.heading_level) * weights['heading_level_factor']
            
            # Content-Type Gewichtung aus zentraler Config
            if settings and hasattr(settings, 'deduplication'):
                content_weights = settings.deduplication.content_type_weights
            else:
                content_weights = DEDUPLICATION_CONFIG['content_type_weights']
            
            # Mapping von ContentType enum zu config keys
            type_mapping = {
                ContentType.LEGAL_TEXT: 'legal_text',
                ContentType.TECHNICAL_SPEC: 'technical_spec',
                ContentType.NARRATIVE: 'narrative',
                ContentType.TABLE: 'table',
                ContentType.LIST: 'list',
                ContentType.MIXED: 'mixed'
            }
            
            config_key = type_mapping.get(chunk.content_type, 'mixed')
            score += content_weights.get(config_key, 25)
            
            # Token count (mehr Inhalt = besser)
            if hasattr(chunk, 'token_count') and chunk.token_count:
                score += chunk.token_count * weights['token_count_factor']
            
            return score
        
        # Wähle Chunk mit höchstem Score
        best_chunk = max(chunks, key=score_chunk)
        return best_chunk

# Demo-Funktion
def demo_semantic_simhash():
    """Demonstriert die erweiterte semantische Deduplication"""
    
    print("🔍 ERWEITERTE SEMANTISCHE SIMHASH DEDUPLICATION")
    print("=" * 55)
    
    # Mock-Chunks für Demo
    from bu_processor.pipeline.chunker import HierarchicalChunk, ContentType, DocumentNode, HeadingType
    
    # Erstelle Test-Chunks mit semantischen Duplikaten
    test_texts = [
        # Original
        "Die Berufsunfähigkeitsversicherung bietet umfassenden Schutz bei Verlust der Arbeitsfähigkeit durch Krankheit oder Unfall.",
        
        # Semantisches Duplikat (andere Wortwahl, gleiche Bedeutung)
        "Eine Berufsunfähigkeitsversicherung gewährt vollständigen Schutz, wenn die Erwerbsfähigkeit durch Erkrankung oder Verletzung verloren geht.",
        
        # Near-Duplikat (ähnlich aber nicht identisch)
        "Die BU-Versicherung schützt vor finanziellen Einbußen bei beruflicher Arbeitsunfähigkeit aufgrund gesundheitlicher Probleme.",
        
        # Unterschiedlicher Inhalt
        "Der Versicherungsnehmer hat Anspruch auf umfassende Beratung durch qualifizierte Fachkräfte bei Vertragsabschluss.",
        
        # Technischer Inhalt (anderer Content-Type)
        "Die automatisierte Dokumentverarbeitung erfolgt durch ML-Algorithmen mit einer Genauigkeit von über 95% bei der Texterkennung."
    ]
    
    chunks = []
    for i, text in enumerate(test_texts):
        mock_node = DocumentNode(
            id=f"node_{i}",
            title=f"Section {i+1}",
            content=text,
            level=1,
            node_type=HeadingType.SECTION,
            position=i * 100
        )
        
        chunk = HierarchicalChunk(
            id=f"chunk_{i}",
            text=text,
            chunk_index=i,
            start_position=i * 200,
            end_position=i * 200 + len(text),
            document_node=mock_node,
            heading_level=1,
            heading_text=f"Test Heading {i+1}",
            full_path=f"Root/Section_{i+1}",
            content_type=ContentType.LEGAL_TEXT if i < 4 else ContentType.TECHNICAL_SPEC,
            token_count=len(text.split()),
            sentence_count=text.count('.') + 1
        )
        chunks.append(chunk)
    
    print(f"📄 Test Chunks: {len(chunks)}")
    
    # Teste semantische Deduplication
    engine = EnhancedDeduplicationEngine()
    deduplicated_chunks, report = engine.deduplicate_chunks_semantically(chunks)
    
    print(f"\n📊 ERGEBNISSE:")
    print(f"   Original Chunks: {report['original_chunks']}")
    print(f"   Deduplizierte Chunks: {report['deduplicated_chunks']}")
    print(f"   Entfernte Chunks: {report['removed_chunks']}")
    print(f"   Deduplication Rate: {report['deduplication_rate']:.1%}")
    print(f"   Gefundene Cluster: {report['clusters_found']}")
    
    print(f"\n🎯 SEMANTISCHE ANALYSE:")
    detector_stats = report['detector_stats']
    print(f"   Verarbeitete Chunks: {detector_stats['chunks_processed']}")
    print(f"   Gefundene Duplikate: {detector_stats['duplicates_found']}")
    print(f"   Semantische Cluster: {detector_stats['semantic_clusters']}")
    
    # Zeige Details der verbliebenen Chunks
    print(f"\n📝 VERBLIEBENE CHUNKS:")
    for i, chunk in enumerate(deduplicated_chunks, 1):
        print(f"   {i}. [{chunk.content_type.value}] {chunk.text[:80]}...")
    
    print(f"\n✅ VERBESSERUNGEN:")
    improvements = [
        "✅ Korrekte SimHash-Implementierung mit Bit-Vektor Akkumulation",
        "✅ N-Gram-basierte semantische Erfassung (1-3 Grams)",
        "✅ Intelligente Text-Normalisierung und Feature-Extraktion",
        "✅ Content-Type-gewichtete Ähnlichkeitsberechnung",
        "✅ Hamming-Distanz für präzise Duplikatserkennung",
        "✅ Kombinierte Hamming + Feature-Similarity Metriken",
        "✅ Quality-aware Representative Selection",
        "✅ Performance-optimiert mit Caching"
    ]
    
    for improvement in improvements:
        print(f"   {improvement}")

if __name__ == "__main__":
    demo_semantic_simhash()
