#!/usr/bin/env python3
"""
ERWEITERTE SIMHASH SEMANTISCHE DEDUPLICATION - REFINED v2.0
==========================================================

Verbesserte SimHash-Implementierung zur Erkennung semantischer Duplikate
mit intelligenter Term-Gewichtung und gewichteter Ähnlichkeitsmetrik.

NEUE FEATURES:
- Corpus-weite TF-IDF-ähnliche Term-Gewichtung
- Intelligente Fachbegriff-Erkennung für Versicherungsdomäne
- Gewichtete kombinierte Ähnlichkeitsmetrik (70% Hamming + 30% Features)
- Erweiterte semantische Feature-Extraktion
- Performance-optimierte Cluster-Bildung
"""

from __future__ import annotations  # postpone annotation eval

import mmh3
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Iterable
from collections import defaultdict, Counter

# Import required types
from .content_types import ContentType  # REQUIRED: brings ContentType into scope
import re
from dataclasses import dataclass
import hashlib
import math
from functools import lru_cache

# Config import
try:
    from ..core.config import settings, DEDUPLICATION_CONFIG
    import structlog
    logger = structlog.get_logger("simhash_semantic_deduplication")
except ImportError:
    # Fallback-Konfigurationen und Logger
    DEDUPLICATION_CONFIG = {
        'simhash_bit_size': 64, 'simhash_ngram_size': 3, 'similarity_threshold': 0.85,
        'hamming_threshold': 8,
        'content_type_weights': {'legal_text': 1.5, 'narrative': 1.0},
        'selection_weights': {'text_length_factor': 0.001, 'heading_level_factor': 10.0},
        'enable_caching': True, 'cache_size_limit': 10000, 'enable_semantic_features': True,
        'min_important_word_length': 6, 'min_token_length': 3
    }
    import logging
    logger = logging.getLogger("simhash_semantic_deduplication")

    from dataclasses import dataclass
    from enum import Enum  # retained for type checking tools
    from .content_types import ContentType

@dataclass
class HierarchicalChunk:
    id: str
    text: str
    content_type: ContentType
    heading_level: Optional[int] = None
    token_count: int = 0

@dataclass
class SimHashResult:
    chunk_id: str
    simhash_value: int
    bit_vector: np.ndarray
    token_count: int
    ngram_count: int
    semantic_features: Dict[str, float]


class SemanticSimHashGenerator:
    """Generate SimHash with semantic understanding."""
    
    def __init__(self, embedding_model: Optional[str] = None, ngram_size: int = 3, bit_size: int = 64):
        self.embedding_model = embedding_model
        self.ngram_size = ngram_size
        self.bit_size = bit_size
        self.encoder = None
        self.stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        self.config = {
            'min_token_length': 3,
            'min_important_word_length': 6,
            'enable_semantic_features': True,
            'cache_size_limit': 10000,
            'similarity_threshold': 0.85,
            'hamming_threshold': 3,
            'content_type_weights': {
                'text': 1.0,
                'heading': 1.5,
                'title': 2.0,
                'footer': 0.5,
                'header': 0.7
            },
            'selection_weights': {
                'similarity': 0.7,
                'position': 0.2,
                'length': 0.1
            }
        }  # Add config attribute
        
        # Add text patterns for semantic analysis
        import re
        self.text_patterns = {
            'legal_terms': re.compile(r'\b(rechtsschutz|versicherung|klausel|bedingung|haftung)\b', re.IGNORECASE),
            'insurance_benefits': re.compile(r'\b(leistung|erstattung|zahlung|deckung|schutz)\b', re.IGNORECASE),
            'medical_terms': re.compile(r'\b(arzt|behandlung|krankenhaus|medizin|therapie)\b', re.IGNORECASE),
            'exclusions': re.compile(r'\b(ausschluss|nicht|kein|ohne|ausgenommen)\b', re.IGNORECASE),
            'numbers': re.compile(r'\d+'),
            'currency': re.compile(r'[€$£¥]\s*\d+|\d+\s*[€$£¥]'),
            'percentages': re.compile(r'\d+\s*%')
        }
        
        if embedding_model:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(embedding_model)
            except ImportError:
                pass
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for processing."""
        return text.lower().strip()
    
    def _extract_features(self, text: str, ngram_size: int) -> List[str]:
        """Extract n-gram features from text."""
        words = text.split()
        if len(words) < ngram_size:
            return [text] if text else ["empty"]
        
        features = []
        for i in range(len(words) - ngram_size + 1):
            features.append(' '.join(words[i:i + ngram_size]))
        return features if features else [text]
    
    def generate(self, text: str) -> int:
        """Generate SimHash for text."""
        if not text:
            return 0
        
        normalized = self._normalize_text(text)
        features = self._extract_features(normalized, self.ngram_size)
        
        # Generate hash
        v = [0] * 64
        for feature in features:
            h = hash(feature)
            for i in range(64):
                if h & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1
        
        simhash = 0
        for i in range(64):
            if v[i] >= 0:
                simhash |= 1 << i
        
        return simhash

    def generate_simhash_for_corpus(self, chunks: List[HierarchicalChunk]) -> Dict[str, SimHashResult]:
        """Generiert SimHashes für eine ganze Liste von Chunks mit Corpus-Analyse."""
        logger.info("starting_corpus_analysis", chunk_count=len(chunks))
        
        ### VERBESSERUNG 2: Corpus-weite TF-IDF-ähnliche Gewichtung ###
        # Sammle alle Tokens aus dem Corpus
        corpus_tokens = []
        for chunk in chunks:
            normalized_text = self._normalize_text(chunk.text)
            tokens = [token for token in normalized_text.split() if token not in self.stopwords]
            corpus_tokens.append(tokens)

        # Berechne Document Frequency für jeden Token
        doc_freq = Counter()
        for doc_tokens in corpus_tokens:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                doc_freq[token] += 1

        num_docs = len(chunks)
        
        # Generiere SimHashes mit IDF-Gewichtung
        results = {}
        for i, chunk in enumerate(chunks):
            # Berechne IDF-Scores für diesen Chunk
            chunk_tokens = corpus_tokens[i]
            token_idf = {}
            for token in set(chunk_tokens):
                # IDF = log(N / df(token))
                idf_score = math.log(num_docs / (doc_freq[token] + 1))
                token_idf[token] = idf_score
            
            # Generiere SimHash mit IDF-Gewichtung
            results[chunk.id] = self._generate_single_simhash(chunk, token_idf)
            
        logger.info("corpus_analysis_completed", unique_terms=len(doc_freq))
        return results

    def _generate_single_simhash(self, chunk: HierarchicalChunk, token_idf: Dict[str, float]) -> SimHashResult:
        """Generiert einen einzelnen SimHash mit vorberechneter IDF-Gewichtung."""
        
        cache_key = f"simhash_{chunk.id}_{hash(chunk.text)}"
        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key]

        semantic_features = self._extract_semantic_features(chunk.text)
        ngrams = self._generate_ngrams(chunk.text)
        
        bit_vector = np.zeros(self.bit_size, dtype=np.float64)
        content_weight = self._get_content_type_weight(chunk.content_type)
        
        for ngram in ngrams:
            ngram_hash = mmh3.hash(ngram, signed=False)
            
            ### VERBESSERUNG 3: Intelligente Term-Gewichtung ###
            weight = self._calculate_term_weight(ngram, semantic_features, token_idf) * content_weight
            
            for bit_pos in range(self.bit_size):
                if ngram_hash & (1 << bit_pos):
                    bit_vector[bit_pos] += weight
                else:
                    bit_vector[bit_pos] -= weight
        
        simhash_value = 0
        for bit_pos in range(self.bit_size):
            if bit_vector[bit_pos] > 0:
                simhash_value |= (1 << bit_pos)
        
        result = SimHashResult(
            chunk_id=chunk.id, simhash_value=simhash_value, bit_vector=bit_vector,
            token_count=len(chunk.text.split()), ngram_count=len(ngrams), semantic_features=semantic_features
        )
        
        if self._cache is not None:
            if len(self._cache) >= self._cache_size_limit:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = result
            
        return result

    @lru_cache(maxsize=1024)
    def _calculate_term_weight_cached(self, term: str, features_hash: str, idf_avg: float) -> float:
        """Cached version of term weight calculation."""
        weight = 1.0
        tokens = term.split()
        
        # IDF-Gewichtung (seltene Terme sind wichtiger)
        weight += idf_avg * 0.5
        
        # Fachbegriff-Bonus
        term_lower = term.lower()
        for pattern_name, pattern in self.text_patterns.items():
            if pattern.search(term):
                if pattern_name == 'legal_terms':
                    weight *= 2.0  # Höchste Priorität für Rechtsbegriffe
                elif pattern_name == 'insurance_benefits':
                    weight *= 1.8  # Leistungsbegriffe
                elif pattern_name in ['medical_terms', 'exclusions']:
                    weight *= 1.6  # Medizin und Ausschlüsse
                else:
                    weight *= 1.3  # Andere Fachbegriffe
        
        # Wichtige Bigramm-Kombinationen
        if len(tokens) == 2:
            for bigram_pattern in self.important_bigrams:
                if re.search(bigram_pattern, term, re.IGNORECASE):
                    weight *= 1.7
                    break
        
        # Längen-basierte Gewichtung (längere Wörter sind spezifischer)
        if tokens:
            avg_len = sum(len(t) for t in tokens) / len(tokens)
            if avg_len > self.config['min_important_word_length']:
                weight *= (1.0 + avg_len * 0.05)
        
        # Numerische Werte sind wichtig für Versicherungen
        if any(pattern.search(term) for pattern in [self.text_patterns['numbers'], 
                                                    self.text_patterns['currency'], 
                                                    self.text_patterns['percentages']]):
            weight *= 1.4
            
        return weight
    
    def _calculate_term_weight(self, term: str, features: Dict[str, float], idf_scores: Dict[str, float]) -> float:
        """Berechnet das intelligente semantische Gewicht eines Terms."""
        tokens = term.split()
        
        # Berechne durchschnittliche IDF für den Term
        if tokens:
            avg_idf = sum(idf_scores.get(token.lower(), 0.0) for token in tokens) / len(tokens)
        else:
            avg_idf = 0.0
        
        # Erstelle Hash für Features (für Caching)
        features_hash = hashlib.md5(str(sorted(features.items())).encode()).hexdigest()[:8]
        
        # Verwende cached Version
        return self._calculate_term_weight_cached(term, features_hash, avg_idf)

    def _generate_ngrams(self, text: str) -> List[str]:
        normalized_text = self._normalize_text(text)
        tokens = [token for token in normalized_text.split() if token not in self.stopwords and len(token) >= self.config['min_token_length']]
        
        if len(tokens) < self.ngram_size:
            return tokens or [normalized_text[:20]]  # Fallback für sehr kurze Texte

        ngrams = []
        
        # Standard N-Grams
        for i in range(len(tokens) - self.ngram_size + 1):
            ngram = ' '.join(tokens[i:i+self.ngram_size])
            ngrams.append(ngram)
        
        # Wichtige Einzelwörter hinzufügen
        for token in tokens:
            if self._is_important_single_token(token):
                ngrams.append(token)
        
        # Wichtige Bigramme hinzufügen
        if len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                bigram = f"{tokens[i]} {tokens[i+1]}"
                if self._is_important_bigram(bigram):
                    ngrams.append(bigram)
        
        return list(set(ngrams))

    @lru_cache(maxsize=1024)
    def _normalize_text(self, text: str) -> str:
        normalized = text.lower().strip()
        # Bewahre wichtige Satzzeichen für Versicherungskontext
        normalized = re.sub(r'[^\w\s.,!?%-]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    def _extract_features(self, text: str, n: int) -> List[Tuple[str, float]]:
        """Extrahiert Features (n-grams) mit Gewichten für SimHash-Berechnung.
        
        Args:
            text: Normalisierter Text 
            n: N-Gram Größe
            
        Returns:
            Liste von (feature, weight) Tupeln
        """
        # Tokenize the text
        tokens = [token for token in text.split() if token not in self.stopwords and len(token) >= self.config['min_token_length']]
        
        if len(tokens) < n:
            # For very short texts, return individual tokens with weight 1.0
            return [(token, 1.0) for token in tokens] if tokens else [(text[:20], 1.0)]

        features = []
        
        # Generate n-grams
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i:i+n])
            # Calculate basic weight (can be enhanced later)
            weight = self._calculate_basic_feature_weight(ngram)
            features.append((ngram, weight))
        
        # Add important single tokens
        for token in tokens:
            if self._is_important_single_token(token):
                weight = self._calculate_basic_feature_weight(token)
                features.append((token, weight))
        
        return features

    def _calculate_basic_feature_weight(self, feature: str) -> float:
        """Berechnet ein Grundgewicht für ein Feature basierend auf Mustern.
        
        Args:
            feature: Das zu bewertende Feature (Token oder N-Gram)
            
        Returns:
            Gewichtsfaktor (standardmäßig 1.0, höher für wichtige Begriffe)
        """
        weight = 1.0
        feature_lower = feature.lower()
        
        # Pattern-basierte Gewichtung
        for pattern_name, pattern in self.text_patterns.items():
            if pattern.search(feature):
                if pattern_name == 'legal_terms':
                    weight *= 2.0
                elif pattern_name == 'insurance_benefits':
                    weight *= 1.8
                elif pattern_name in ['medical_terms', 'exclusions']:
                    weight *= 1.6
                else:
                    weight *= 1.3
        
        # Längenbonus für spezifische Begriffe
        tokens = feature.split()
        if tokens:
            avg_len = sum(len(t) for t in tokens) / len(tokens)
            if avg_len > self.config['min_important_word_length']:
                weight *= (1.0 + avg_len * 0.05)
        
        return weight

    def _is_important_single_token(self, token: str) -> bool:
        if len(token) < self.config['min_token_length']: 
            return False
        if any(pattern.search(token) for pattern in self.text_patterns.values()): 
            return True
        return len(token) >= self.config['min_important_word_length']

    def _is_important_bigram(self, bigram: str) -> bool:
        return any(re.search(pattern, bigram, re.IGNORECASE) for pattern in self.important_bigrams)

    def _get_content_type_weight(self, content_type: ContentType) -> float:
        return self.config['content_type_weights'].get(content_type.value, 1.0)
        
    @lru_cache(maxsize=512)
    def _extract_semantic_features(self, text: str) -> Dict[str, float]:
        """Erweiterte semantische Feature-Extraktion."""
        features = {}
        words = text.lower().split()
        word_count = len(words)
        if word_count == 0: 
            return features

        features['word_count'] = word_count
        features['char_count'] = len(text)
        features['avg_word_length'] = sum(len(w) for w in words) / word_count
        
        # Pattern-basierte Dichten
        for name, pattern in self.text_patterns.items():
            matches = pattern.findall(text)
            features[f'{name}_count'] = len(matches)
            features[f'{name}_density'] = len(matches) / word_count if word_count > 0 else 0.0
        
        # Lexikalische Diversität
        unique_words = set(words)
        features['lexical_diversity'] = len(unique_words) / word_count
        
        # Satz-Struktur
        sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))
        features['sentence_count'] = sentence_count
        features['words_per_sentence'] = word_count / sentence_count
        
        # Semantische Komplexität
        features['semantic_complexity'] = (
            features['lexical_diversity'] * 0.3 +
            features.get('legal_terms_density', 0) * 0.4 +
            features.get('insurance_benefits_density', 0) * 0.3
        )
        
        return features


class SemanticDuplicateDetector:
    """Erweiterte Duplikatserkennung mit gewichteter Ähnlichkeitsmetrik."""
    
    def __init__(self, similarity_threshold: Optional[float] = None, hamming_threshold: Optional[int] = None):
        self.config = DEDUPLICATION_CONFIG
        self.similarity_threshold = similarity_threshold or self.config['similarity_threshold']
        self.hamming_threshold = hamming_threshold or self.config['hamming_threshold']
        self.simhash_generator = SemanticSimHashGenerator()
        self._stats = defaultdict(int)

    def find_semantic_duplicates(self, chunks: List[HierarchicalChunk]) -> Dict[str, List[str]]:
        if not chunks: 
            return {}
        
        logger.info("semantic_duplicate_detection_started", chunk_count=len(chunks))
        
        ### VERBESSERUNG 5: Corpus-weite Analyse ###
        simhash_results = self.simhash_generator.generate_simhash_for_corpus(chunks)
        self._stats['chunks_processed'] = len(simhash_results)
        
        # Hamming-basierte Vorauswahl
        pre_clusters = self._find_hamming_clusters(simhash_results)
        
        ### VERBESSERUNG 6: Semantische Cluster-Verfeinerung ###
        refined_clusters = self._refine_clusters_semantically(pre_clusters, simhash_results)
        
        self._stats['duplicates_found'] = sum(len(cluster) - 1 for cluster in refined_clusters.values())
        self._stats['semantic_clusters'] = len(refined_clusters)
        
        logger.info("semantic_duplicate_detection_completed", **self._stats)
        return refined_clusters

    def _find_hamming_clusters(self, simhash_results: Dict[str, SimHashResult]) -> List[List[str]]:
        """Findet potenzielle Duplikat-Cluster basierend auf Hamming-Distanz."""
        clusters = []
        processed_indices = set()
        chunk_ids = list(simhash_results.keys())
        
        for i in range(len(chunk_ids)):
            if i in processed_indices:
                continue
            
            current_cluster = [chunk_ids[i]]
            processed_indices.add(i)
            hash1 = simhash_results[chunk_ids[i]].simhash_value
            
            for j in range(i + 1, len(chunk_ids)):
                if j in processed_indices:
                    continue
                
                hash2 = simhash_results[chunk_ids[j]].simhash_value
                if self._calculate_hamming_distance(hash1, hash2) <= self.hamming_threshold:
                    current_cluster.append(chunk_ids[j])
                    processed_indices.add(j)
            
            if len(current_cluster) > 1:
                clusters.append(current_cluster)
        return clusters
    
    @lru_cache(maxsize=2048)
    def _calculate_hamming_distance(self, hash1: int, hash2: int) -> int:
        return bin(hash1 ^ hash2).count('1')

    def _refine_clusters_semantically(self, clusters: List[List[str]], simhash_results: Dict[str, SimHashResult]) -> Dict[str, List[str]]:
        """Verfeinert Cluster durch gewichtete semantische Ähnlichkeitsprüfung."""
        final_clusters = {}
        cluster_id_counter = 0

        for cluster in clusters:
            # Graph-basierte Cluster-Verfeinerung
            similarity_graph = defaultdict(list)
            
            # Berechne paarweise Ähnlichkeiten
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    chunk_id1, chunk_id2 = cluster[i], cluster[j]
                    
                    ### VERBESSERUNG 7: Gewichtete kombinierte Ähnlichkeit ###
                    similarity = self._calculate_combined_similarity(
                        simhash_results[chunk_id1],
                        simhash_results[chunk_id2]
                    )
                    
                    if similarity >= self.similarity_threshold:
                        similarity_graph[chunk_id1].append((chunk_id2, similarity))
                        similarity_graph[chunk_id2].append((chunk_id1, similarity))
            
            # Finde verbundene Komponenten
            if similarity_graph:
                visited = set()
                for chunk_id in cluster:
                    if chunk_id not in visited:
                        component = self._extract_connected_component(chunk_id, similarity_graph, visited)
                        if len(component) > 1:
                            final_clusters[f"cluster_{cluster_id_counter}"] = component
                            cluster_id_counter += 1
        
        return final_clusters
    
    def _extract_connected_component(self, start_node: str, graph: dict, visited: set) -> List[str]:
        """Extrahiert eine verbundene Komponente aus dem Ähnlichkeitsgraph."""
        component = []
        queue = [start_node]
        visited.add(start_node)
        
        while queue:
            node = queue.pop(0)
            component.append(node)
            
            for neighbor, similarity in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return component

    def _calculate_combined_similarity(self, result1: SimHashResult, result2: SimHashResult) -> float:
        """Berechnet gewichteten kombiniereten Ähnlichkeits-Score (70% Hamming + 30% Features)."""
        
        # Hamming-Ähnlichkeit (strukturell)
        hamming_dist = self._calculate_hamming_distance(result1.simhash_value, result2.simhash_value)
        hamming_similarity = 1.0 - (hamming_dist / self.simhash_generator.bit_size)
        
        # Feature-Ähnlichkeit (semantisch)
        feature_similarity = self._calculate_feature_similarity(
            result1.semantic_features, result2.semantic_features
        )
        
        ### VERBESSERUNG 8: Gewichteter kombinierter Score ###
        # 70% strukturell (Hamming), 30% semantisch (Features)
        combined_similarity = (hamming_similarity * 0.7) + (feature_similarity * 0.3)
        
        return combined_similarity

    def _calculate_feature_similarity(self, features1: Dict, features2: Dict) -> float:
        """Berechnet die erweiterte Kosinus-Ähnlichkeit der semantischen Features."""
        
        # Alle verfügbaren Feature-Keys sammeln
        all_keys = set(features1.keys()) | set(features2.keys())
        if not all_keys:
            return 0.0
        
        # Feature-Vektoren erstellen
        vec1 = np.array([features1.get(key, 0.0) for key in all_keys])
        vec2 = np.array([features2.get(key, 0.0) for key in all_keys])
        
        # Kosinus-Ähnlichkeit berechnen
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        
        # Gewichtung wichtiger Features
        important_features = ['legal_terms_density', 'insurance_benefits_density', 'semantic_complexity']
        important_similarity = 0.0
        important_count = 0
        
        for feature in important_features:
            if feature in features1 and feature in features2:
                # Manhattan-Distanz für wichtige Features
                diff = abs(features1[feature] - features2[feature])
                important_similarity += (1.0 - diff)
                important_count += 1
        
        if important_count > 0:
            important_similarity /= important_count
            # Kombiniere Standard-Kosinus mit wichtigen Features (60/40)
            cosine_sim = cosine_sim * 0.6 + important_similarity * 0.4
        
        return max(0.0, cosine_sim)

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)


class EnhancedDeduplicationEngine:
    """Erweiterte Deduplication-Engine mit semantischem SimHash v2.0."""
    
    def __init__(self, similarity_threshold: Optional[float] = None):
        self.config = DEDUPLICATION_CONFIG
        self.similarity_threshold = similarity_threshold or self.config['similarity_threshold']
        self.selection_weights = self.config['selection_weights']
        
        self.semantic_detector = SemanticDuplicateDetector(
            similarity_threshold=self.similarity_threshold,
            hamming_threshold=self.config.get('hamming_threshold', 8)
        )
        
        logger.info("enhanced_deduplication_engine_v2_initialized", 
                   similarity_threshold=self.similarity_threshold)

    def deduplicate_chunks_semantically(self, chunks: List[HierarchicalChunk]) -> Tuple[List[HierarchicalChunk], Dict[str, any]]:
        """Hauptmethode für erweiterte semantische Deduplication."""
        
        if not chunks:
            return chunks, {'error': 'No chunks provided'}
        
        logger.info("enhanced_semantic_deduplication_started", original_chunks=len(chunks))
        
        # Semantische Duplikate finden
        duplicate_clusters = self.semantic_detector.find_semantic_duplicates(chunks)
        
        # Chunk-Lookup erstellen
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        chunks_to_remove = set()
        
        # Cluster verarbeiten und beste Repräsentanten wählen
        cluster_stats = []
        for cluster_id, chunk_ids in duplicate_clusters.items():
            if len(chunk_ids) < 2:
                continue
            
            cluster_chunks = [chunks_by_id[cid] for cid in chunk_ids if cid in chunks_by_id]
            representative = self._select_best_representative(cluster_chunks)
            
            # Andere Chunks zur Entfernung markieren
            removed_in_cluster = []
            for chunk_id in chunk_ids:
                if chunk_id != representative.id:
                    chunks_to_remove.add(chunk_id)
                    removed_in_cluster.append(chunk_id)
            
            cluster_stats.append({
                'cluster_id': cluster_id,
                'total_chunks': len(chunk_ids),
                'representative': representative.id,
                'removed_chunks': removed_in_cluster
            })
            
            logger.debug("cluster_processed", cluster_id=cluster_id, 
                        chunk_count=len(chunk_ids), representative=representative.id)
        
        # Deduplizierte Liste erstellen
        deduplicated_chunks = [chunk for chunk in chunks if chunk.id not in chunks_to_remove]
        
        # Detaillierter Report
        deduplication_rate = len(chunks_to_remove) / len(chunks) if len(chunks) > 0 else 0.0
        report = {
            'original_chunks': len(chunks),
            'deduplicated_chunks': len(deduplicated_chunks),
            'removed_chunks': len(chunks_to_remove),
            'deduplication_rate': deduplication_rate,
            'clusters_found': len(duplicate_clusters),
            'detector_stats': self.semantic_detector.get_stats(),
            'cluster_details': cluster_stats,
            'performance_metrics': {
                'similarity_threshold': self.similarity_threshold,
                'hamming_threshold': self.config.get('hamming_threshold', 8),
                'processing_method': 'corpus_wide_tfidf'
            }
        }
        
        logger.info("enhanced_semantic_deduplication_completed", **{k: v for k, v in report.items() if k != 'cluster_details'})
        
        return deduplicated_chunks, report

    def _select_best_representative(self, chunks: List[HierarchicalChunk]) -> HierarchicalChunk:
        """Wählt den besten Chunk als Repräsentant für einen Cluster."""
        
        if len(chunks) == 1:
            return chunks[0]
        
        def score_chunk(chunk):
            score = 0.0
            
            # Textlänge (längere Texte sind oft informativer)
            score += len(chunk.text) * self.selection_weights['text_length_factor']
            
            # Hierarchie-Level (wichtigere Überschriften)
            if hasattr(chunk, 'heading_level') and chunk.heading_level:
                score += (7 - chunk.heading_level) * self.selection_weights['heading_level_factor']
            
            # Content-Type Bonus
            content_weights = self.config['content_type_weights']
            type_mapping = {
                ContentType.LEGAL_TEXT: 'legal_text',
                ContentType.TECHNICAL_SPEC: 'technical_spec', 
                ContentType.NARRATIVE: 'narrative',
                ContentType.TABLE: 'table',
                ContentType.LIST: 'list',
                ContentType.MIXED: 'mixed'
            }
            config_key = type_mapping.get(chunk.content_type, 'mixed')
            score += content_weights.get(config_key, 1.0) * 25
            
            # Token-Count Bonus
            if hasattr(chunk, 'token_count') and chunk.token_count:
                score += chunk.token_count * self.selection_weights.get('token_count_factor', 0.1)
            
            # Fachbegriff-Dichte Bonus
            for pattern_name, pattern in self.semantic_detector.simhash_generator.text_patterns.items():
                matches = len(pattern.findall(chunk.text))
                if matches > 0:
                    if pattern_name == 'legal_terms':
                        score += matches * 10
                    elif pattern_name == 'insurance_benefits':
                        score += matches * 8
                    else:
                        score += matches * 5
            
            return score
        
        best_chunk = max(chunks, key=score_chunk)
        return best_chunk


#############################################
# Backwards compatible lightweight wrappers #
#############################################

def calculate_simhash(text: str, *, bit_size: Optional[int] = None, ngram_size: Optional[int] = None) -> int:
    """Backward-compatible helper expected by older tests.

    Generates a simple simhash for a single text using the current
    SemanticSimHashGenerator internals. Only the integer hash value is
    returned (legacy behaviour).
    """
    generator = SemanticSimHashGenerator(ngram_size=ngram_size or 3, bit_size=bit_size or 64)
    # We re-use private normalization/tokenization for consistency.
    features = generator._extract_features(generator._normalize_text(text), generator.ngram_size)
    # Basic SimHash logic (duplicated minimal subset to avoid building HierarchicalChunk).
    bit_accumulator = [0] * generator.bit_size
    for feature, weight in features:
        token_hash = mmh3.hash(feature.encode('utf-8'), 42, signed=False) & ((1 << generator.bit_size) - 1)
        for i in range(generator.bit_size):
            bit = 1 if (token_hash >> i) & 1 else 0
            bit_accumulator[i] += weight if bit else -weight
    simhash_value = 0
    for i, acc in enumerate(bit_accumulator):
        if acc >= 0:
            simhash_value |= (1 << i)
    return simhash_value


def find_duplicates(documents: Iterable[Dict[str, str]], threshold: int = 8) -> Dict[str, Set[str]]:
    """Naive duplicate finder for backwards compatibility tests.

    Uses simple pairwise Hamming distance on legacy simhash values.
    Returns mapping from a representative doc id to a set of duplicate ids.
    """
    docs = list(documents)
    if not docs:
        return {}
    hashes: Dict[str, int] = {d['id']: calculate_simhash(d.get('text', '')) for d in docs}
    # Pairwise comparison (O(n^2)) is acceptable for small test corpora.
    def hamming(a: int, b: int) -> int:
        return (a ^ b).bit_count()
    clusters: Dict[str, Set[str]] = {}
    ids = list(hashes.keys())
    for i, id_a in enumerate(ids):
        for id_b in ids[i+1:]:
            if hamming(hashes[id_a], hashes[id_b]) <= threshold:
                # choose stable representative (first seen)
                rep = id_a
                clusters.setdefault(rep, set()).add(id_b)
    return clusters

# NOTE: Former demo_enhanced_semantic_simhash() removed to slim production module.
    for i, id_a in enumerate(ids):
        for id_b in ids[i+1:]:
            if hamming(hashes[id_a], hashes[id_b]) <= threshold:
                # choose stable representative (first seen)
                rep = id_a
                clusters.setdefault(rep, set()).add(id_b)
    return clusters

# NOTE: Former demo_enhanced_semantic_simhash() removed to slim production module.
