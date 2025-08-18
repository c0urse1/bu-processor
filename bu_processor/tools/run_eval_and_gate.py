import sys, json, tempfile
from bu_processor.eval.harness import run_eval, save_report
from bu_processor.eval.quality_gate import passes, explain

CORPUS = [
    {"title":"Insurance A","source":"eval","chunks":[
        {"text":"Professional liability insurance covers financial losses from negligence.", "section":"Insurance","page":1},
        {"text":"Pet insurance may include vet visits.", "section":"Insurance","page":2},
    ]},
    {"title":"Finance A","source":"eval","chunks":[
        {"text":"Corporate finance optimizes capital structure and funding.", "section":"Finance","page":1},
    ]},
    {"title":"Pets A","source":"eval","chunks":[
        {"text":"Cats are small domestic animals and love to sleep.", "section":"Pets","page":1},
    ]},
]

GOLDEN = [
    {"query":"Which insurance covers financial loss from negligence?",
     "gold_doc_ids": [],  # left empty (doc_id is generated); rely on keywords + retrieval ids
     "answer_keywords": ["insurance","negligence","financial"]},
    {"query":"What optimizes capital structure for companies?",
     "answer_keywords": ["corporate","finance","capital"]},
]

THRESHOLDS = {"hit@3": 0.66, "mrr": 0.50, "citation_acc": 0.80, "faithfulness": 0.66}

def main():
    dburl = "sqlite:///" + str(tempfile.gettempdir()) + "/eval_gate.db"
    report = run_eval(dburl, CORPUS, GOLDEN, top_k_retrieve=5)
    save_report(report, "eval_report.json")
    ok = passes(report["aggregate"], THRESHOLDS)
    print(json.dumps(report["aggregate"], indent=2))
    print(explain(report["aggregate"], THRESHOLDS))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
