from bu_processor.eval.metrics import (
    hit_at_k, reciprocal_rank, aggregate_metrics, 
    citation_accuracy, faithfulness_keywords
)

def test_hit_at_k():
    retrieved = ["doc1", "doc2", "doc3"]
    gold = ["doc2", "doc4"]
    
    assert hit_at_k(retrieved, gold, 1) == 0.0  # doc2 not in top 1
    assert hit_at_k(retrieved, gold, 2) == 1.0  # doc2 in top 2
    assert hit_at_k(retrieved, gold, 3) == 1.0  # doc2 in top 3

def test_reciprocal_rank():
    retrieved = ["doc1", "doc2", "doc3"]
    gold = ["doc2", "doc4"]
    
    assert reciprocal_rank(retrieved, gold) == 0.5  # doc2 at position 2, so 1/2 = 0.5
    
    retrieved = ["doc4", "doc1", "doc2"]
    assert reciprocal_rank(retrieved, gold) == 1.0  # doc4 at position 1, so 1/1 = 1.0

def test_aggregate_metrics():
    rows = [
        {"hit@1": 1.0, "hit@3": 1.0, "mrr": 1.0},
        {"hit@1": 0.0, "hit@3": 1.0, "mrr": 0.5},
    ]
    
    agg = aggregate_metrics(rows)
    assert agg["hit@1"] == 0.5  # (1.0 + 0.0) / 2
    assert agg["hit@3"] == 1.0  # (1.0 + 1.0) / 2
    assert agg["mrr"] == 0.75   # (1.0 + 0.5) / 2

def test_citation_accuracy():
    # Good case: all paragraphs have valid citations
    answer = "Insurance covers losses. [1]\n\nFinance helps companies. [2]"
    sources = [{"id": "chunk1"}, {"id": "chunk2"}]
    assert citation_accuracy(answer, sources) == 1.0
    
    # Mixed case: one paragraph missing citation
    answer = "Insurance covers losses. [1]\n\nFinance helps companies."
    assert citation_accuracy(answer, sources) == 0.5
    
    # Bad case: invalid citation index
    answer = "Insurance covers losses. [1]\n\nFinance helps companies. [5]"
    assert citation_accuracy(answer, sources) == 0.5  # only first paragraph valid

def test_faithfulness_keywords():
    answer = "Professional liability insurance covers financial losses."
    cited_texts = ["Professional insurance helps with liability and financial matters."]
    keywords = ["insurance", "financial"]
    
    # Both keywords in answer and context
    assert faithfulness_keywords(answer, cited_texts, keywords) == 1.0
    
    # Keywords only in answer
    cited_texts = ["Some other text about pets."]
    assert faithfulness_keywords(answer, cited_texts, keywords) == 0.5
    
    # No keywords match
    answer = "Cats are cute pets."
    assert faithfulness_keywords(answer, cited_texts, keywords) == 0.0
