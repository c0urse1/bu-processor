from bu_processor.eval.harness import run_eval

def test_eval_harness_deterministic(tmp_path):
    dburl = f"sqlite:///{tmp_path/'eval.db'}"
    corpus = [
        {"title":"Doc1","source":"eval","chunks":[
            {"text":"Professional liability insurance covers financial losses from negligence.","section":"Insurance","page":1},
            {"text":"Pet insurance may include vet visits.","section":"Insurance","page":2},
        ]},
        {"title":"Doc2","source":"eval","chunks":[
            {"text":"Corporate finance optimizes capital structure and funding.","section":"Finance","page":1},
        ]},
    ]
    golden = [
        {"query":"Which insurance covers financial loss from negligence?",
         "answer_keywords":["insurance","negligence","financial"]},
        {"query":"What optimizes capital structure for companies?",
         "answer_keywords":["finance","capital"]},
    ]
    report = run_eval(dburl, corpus, golden, top_k_retrieve=5)
    agg = report["aggregate"]
    # sanity checks (not too strict to avoid flakiness)
    assert agg["hit@3"] >= 0.5
    assert agg["citation_acc"] >= 0.5
