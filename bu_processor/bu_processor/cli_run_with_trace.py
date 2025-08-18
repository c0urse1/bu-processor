from bu_processor.telemetry.trace import Trace, TraceLogger
from bu_processor.telemetry.wrap import traced_retrieve, traced_pack
from bu_processor.factories import make_hybrid_retriever, make_answerer
from bu_processor.answering.context_packer import pack_context
from bu_processor.answering.synthesize import synthesize_answer

def main(query: str):
    tracer = Trace()
    tlog = TraceLogger(file_path=None)  # add a file path if you want

    retriever = make_hybrid_retriever()
    answerer  = make_answerer()

    hits = traced_retrieve(retriever, query, tracer, final_top_k=8)
    ctx, sources = traced_pack(pack_context, hits, tracer, token_budget=800, sentence_overlap=1, prefer_summary=True)

    with tracer.stage("answer"):
        result = answerer.answer(query=query, packed_context=ctx, sources_table=sources)
    tracer.event("answer.result",
                 text_len=len(result.text),
                 citations=len(result.citations),
                 sources=len(result.sources_table))

    tlog.log(tracer, extra={"query": query})

if __name__ == "__main__":
    main("Which insurance covers financial loss from negligence?")
