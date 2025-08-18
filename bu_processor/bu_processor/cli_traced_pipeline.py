"""
Example CLI that demonstrates end-to-end tracing with the complete pipeline.
"""
from bu_processor.telemetry.trace import Trace, TraceLogger
from bu_processor.telemetry.wrap import traced_retrieve, traced_pack
from bu_processor.factories import make_hybrid_retriever, make_answerer
from bu_processor.answering.context_packer import pack_context
from bu_processor.answering.synthesize import synthesize_answer

def handle_user_question(query: str):
    """
    Complete end-to-end Q&A with tracing.
    """
    tracer = Trace()
    tlog = TraceLogger(file_path="traces.jsonl")  # optional file
    retriever = make_hybrid_retriever()
    answerer  = make_answerer()

    hits = traced_retrieve(retriever, query, tracer, final_top_k=8)
    ctx, sources = traced_pack(pack_context, hits, tracer, token_budget=1000, sentence_overlap=1, prefer_summary=True)

    with tracer.stage("synthesize"):
        result = synthesize_answer(query=query, hits=hits, answerer=answerer, token_budget=1000)

    tracer.event("final", answer_len=len(result.text), citations=len(result.citations))
    tlog.log(tracer, extra={"query": query})
    return result

def main():
    # Example usage
    questions = [
        "Which insurance covers financial loss from negligence?",
        "What are the benefits of corporate finance optimization?",
        "How do cats behave as domestic pets?"
    ]
    
    print("🔍 Running traced Q&A pipeline...")
    for i, q in enumerate(questions, 1):
        print(f"\n--- Question {i}: {q}")
        try:
            result = handle_user_question(q)
            print(f"✅ Answer: {result.text[:100]}...")
            print(f"📊 Citations: {len(result.citations)}, Sources: {len(result.sources_table)}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✨ Check traces.jsonl for detailed execution logs!")

if __name__ == "__main__":
    main()
