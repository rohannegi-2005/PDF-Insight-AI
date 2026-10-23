"""
test_pipeline.py
------------------
Checks the whole project.

    python test_pipeline.py offline    -> offline checks only: no internet,
                                          no AstraDB, no Groq (about 10 seconds)
    python test_pipeline.py            -> offline checks, then a LIVE run on
                                          your 3 sample papers (needs AstraDB + Groq)

OFFLINE part: every fix is tested with fake data and fake LLMs, including
the agent loop (retry, early stop, no evidence, a failing sub-question).
LIVE part: the real pipeline end to end. LLM answers vary from run to
run, so those checks are about structure, not exact wording.
"""

import os
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

import config

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    line = ("  PASS  " if ok else "  FAIL  ") + name
    if not ok and detail:
        line += f"   [{detail}]"
    print(line)


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def make_doc(i, paper="A.pdf", page=1, text=None):
    return Document(
        page_content=text or f"chunk number {i} text",
        metadata={"chunk_id": f"c{i}", "paper_id": paper, "source_file": paper, "page": page},
    )


def ids(docs):
    return [d.metadata["chunk_id"] for d in docs]


# ===========================================================================
# OFFLINE 1: sparse store
# ===========================================================================
def test_sparse_store():
    from core.sparse_store import SparseStore

    section("1. SparseStore (BM25 keyword search)")

    store = SparseStore()
    store.build([])
    check("empty corpus does not crash, store is 'not ready'", not store.is_ready())

    docs = [
        Document(page_content=text, metadata={"paper_id": paper, "page": 1})
        for text, paper in [
            ("apples and oranges grow on trees", "fruit.pdf"),
            ("cars and trucks drive on roads", "vehicles.pdf"),
            ("birds fly in the sky", "fruit.pdf"),
        ]
    ]
    store.build(docs)
    check("store is ready after build()", store.is_ready())
    check("a query with no matching word returns nothing", store.search("zzzz qqqq", 3) == [])
    check("a punctuation-only query returns nothing", store.search("???", 3) == [])
    top = store.search("trucks on roads", 2)
    check("best match comes first", bool(top) and "trucks" in top[0].page_content)
    check("paper_ids are unique, in order", store.paper_ids() == ["fruit.pdf", "vehicles.pdf"])

    # BM25 gives a word zero/negative weight if it is in most chunks. Real
    # matches must still be returned in that case (small collections!).
    tiny = SparseStore()
    tiny.build([Document(page_content="hybrid retrieval with BM25", metadata={}),
                Document(page_content="dense retrieval with embeddings", metadata={})])
    check("a word found in EVERY chunk still returns those chunks", len(tiny.search("retrieval", 5)) == 2)
    single = SparseStore()
    single.build([Document(page_content="hybrid retrieval with BM25", metadata={})])
    check("a one-chunk collection still returns its match", len(single.search("hybrid retrieval", 5)) == 1)


# ===========================================================================
# OFFLINE 2: semantic chunker
# ===========================================================================
class TopicEmbeddings(Embeddings):
    """Fake embeddings: apple sentences, car sentences and 'other' get 3 different directions."""

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "apple" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "car" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_chunker():
    from core.semantic_chunker import SemanticChunker

    section("2. SemanticChunker")

    chunker = SemanticChunker(TopicEmbeddings(), 0.5, 300, 1500)
    apples = [f"Apples are a fruit number {i} that grows on trees in many countries around the world." for i in range(8)]
    cars = [f"The car number {i} drives quickly along the long road every single morning." for i in range(8)]

    chunks = chunker.split_text(" ".join(apples + cars))
    check("two topics give at least 2 chunks", len(chunks) >= 2, f"got {len(chunks)}")
    check(
        "no chunk mixes the two topics",
        all(not ("Apples" in c and "car number" in c) for c in chunks),
    )
    check("every chunk is within the max size", all(len(c) <= 1500 for c in chunks))

    huge = " ".join(["word"] * 800)  # 3999 characters, no sentence breaks
    huge_chunks = chunker.split_text(huge)
    check("text with no sentence breaks is cut at the max size", len(huge_chunks) > 1 and all(len(c) <= 1500 for c in huge_chunks), str([len(c) for c in huge_chunks]))

    check("empty text gives no chunks", chunker.split_text("   ") == [])

    tail = chunker.split_text(" ".join(apples + ["The car stopped."]))
    check("a tiny last chunk is merged into the one before", len(tail) == 1 and tail[0].endswith("The car stopped."), str([len(c) for c in tail]))


# ===========================================================================
# OFFLINE 2b: PDF processing (paper names, stable ids)
# ===========================================================================
def test_document_utils():
    import shutil
    import tempfile

    from core.document_utils import SmartPDFProcessor

    section("2b. SmartPDFProcessor (paper names, stable chunk ids)")

    pdf = "data/sample_research_paper.pdf"
    if not os.path.exists(pdf):
        print("  (skipped: data/sample_research_paper.pdf not found)")
        return

    processor = SmartPDFProcessor(embeddings=TopicEmbeddings(), chunk_method="semantic")
    chunks = processor.process_pdf(pdf, paper_id="my_paper.pdf")

    check("chunks were created", len(chunks) > 0)
    required = ("paper_id", "source_file", "chunk_id", "page")
    check("every chunk has paper_id, source_file, chunk_id and page", all(all(k in c.metadata for k in required) for c in chunks))
    check(
        "paper_id is used as the paper name everywhere",
        all(c.metadata["paper_id"] == "my_paper.pdf" and c.metadata["source_file"] == "my_paper.pdf" for c in chunks),
    )
    check("no chunk is longer than the max size", max(len(c.page_content) for c in chunks) <= config.SEMANTIC_MAX_CHUNK_CHARS)

    temp_dir = tempfile.mkdtemp()
    try:
        temp_copy = os.path.join(temp_dir, "tmpa8f3kd2x.pdf")  # like the random file Streamlit creates
        shutil.copy(pdf, temp_copy)
        again = processor.process_pdf(temp_copy, paper_id="my_paper.pdf")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    check(
        "same paper_id gives the SAME chunk ids, even from a temp file",
        [c.metadata["chunk_id"] for c in again] == [c.metadata["chunk_id"] for c in chunks],
    )

    try:
        processor.process_pdf("data/does_not_exist.pdf")
        check("a missing or broken PDF gives a readable error", False)
    except ValueError as error:
        check("a missing or broken PDF gives a readable error", "Could not read the PDF" in str(error))


# ===========================================================================
# OFFLINE 3: fusion + evidence merging
# ===========================================================================
def test_fusion_and_merge():
    from core.retrieval import reciprocal_rank_fusion
    from graph.workflow import merge_evidence

    section("3. Fusion (RRF) and evidence merging")

    d1, d2, d3, d4 = (make_doc(i) for i in (1, 2, 3, 4))
    fused = reciprocal_rank_fusion([[d1, d2, d3], [d2, d4]])
    check("a chunk found by BOTH retrievers ranks first", ids(fused)[0] == "c2", str(ids(fused)))
    check("fusion has no duplicates", len(fused) == 4)

    merged, new_count = merge_evidence([d1, d2], [d2, d3, d3, d4], cap=3)
    check("merge: known and repeated chunks are not added twice", new_count == 2, f"new_count={new_count}")
    check("merge: newest chunks first, then old, cut at the cap", ids(merged) == ["c3", "c4", "c1"], str(ids(merged)))


def test_hybrid_fallback():
    from core.retrieval import hybrid_search
    from core.sparse_store import SparseStore
    from core.vector_store import get_retriever

    section("3b. Dense retriever settings and hybrid search when AstraDB has a problem")

    captured = {}

    class FakeStore:
        def as_retriever(self, **kwargs):
            captured.update(kwargs)

    get_retriever(FakeStore(), k=8)
    settings = captured["search_kwargs"]
    check("get_retriever: fetch_k is larger than k", settings["fetch_k"] > settings["k"], str(settings))
    check("get_retriever: lambda_mult comes from config", settings["lambda_mult"] == config.MMR_LAMBDA_MULT)

    sparse = SparseStore()
    sparse.build([Document(page_content="hybrid retrieval with BM25", metadata={"chunk_id": "s1", "paper_id": "A.pdf", "page": 1})])

    class BrokenRetriever:
        def invoke(self, query):
            raise RuntimeError("503 Service Unavailable")

    with patch("core.retrieval.get_retriever", return_value=BrokenRetriever()), patch("core.retrieval.time.sleep"):
        results = hybrid_search("hybrid retrieval", object(), sparse)
        check("dense keeps failing -> continues with sparse results only", ids(results) == ["s1"], str(ids(results)))
        try:
            hybrid_search("hybrid retrieval", object(), SparseStore())
            check("dense fails and no sparse index -> clear error", False)
        except RuntimeError as error:
            check("dense fails and no sparse index -> clear error", "no sparse index" in str(error))

    class FlakyRetriever:
        def __init__(self):
            self.calls = 0

        def invoke(self, query):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("timeout")
            return [make_doc(1, text="dense hit")]

    flaky = FlakyRetriever()
    with patch("core.retrieval.get_retriever", return_value=flaky), patch("core.retrieval.time.sleep"):
        results = hybrid_search("hybrid retrieval", object(), sparse)
        check(
            "dense fails once -> retried, then fused with sparse",
            flaky.calls == 2 and set(ids(results)) == {"c1", "s1"},
            f"calls={flaky.calls}, ids={ids(results)}",
        )


# ===========================================================================
# OFFLINE 4: structured-output helper, planner fallback, critic, get_llm
# ===========================================================================
class FlakyLLM:
    """Fake LLM whose 'fixed form' call fails `fail_times` times, then works."""

    def __init__(self, fail_times, result):
        self.fail_times, self.result, self.calls = fail_times, result, 0

    def with_structured_output(self, schema, method=None):
        def run(_):
            self.calls += 1
            if self.calls <= self.fail_times:
                raise RuntimeError("Tool choice is required, but model did not call a tool")
            return self.result

        return RunnableLambda(run)


def test_agents():
    from agents.critic_agent import EvidenceVerdict, evaluate_evidence
    from agents.prompts import CRITIC_PROMPT, DECOMPOSE_PROMPT
    from agents.research_agent import SubQuestionPlan, decompose_question
    from agents.structured import invoke_structured

    section("4. Agents: structured output, planner, critic, LLM setup")

    inputs = {"question": "q", "evidence": "e", "papers": "- A.pdf"}
    verdict = EvidenceVerdict(sufficient=True, reason="ok")

    flaky = FlakyLLM(fail_times=1, result=verdict)
    got = invoke_structured(CRITIC_PROMPT, EvidenceVerdict, inputs, attempts=2, llm=flaky)
    check("structured call retries after one failure and succeeds", got.sufficient is True and flaky.calls == 2)

    broken = FlakyLLM(fail_times=99, result=verdict)
    try:
        invoke_structured(CRITIC_PROMPT, EvidenceVerdict, inputs, attempts=2, llm=broken)
        check("structured call raises after all attempts fail", False)
    except RuntimeError:
        check("structured call raises after all attempts fail", broken.calls == 2)

    with patch("agents.research_agent.invoke_structured", side_effect=RuntimeError("boom")):
        check("planner failure falls back to the original question", decompose_question("big q") == ["big q"])
    with patch("agents.research_agent.invoke_structured", return_value=SubQuestionPlan(sub_questions=["a", "b", "c", "d", "e"])):
        check("planner result is cut to MAX sub-questions", decompose_question("q", max_sub_questions=3) == ["a", "b", "c"])
    with patch("agents.research_agent.invoke_structured", return_value=SubQuestionPlan(sub_questions=[])):
        check("empty planner result falls back to the original question", decompose_question("q") == ["q"])
    with patch("agents.research_agent.invoke_structured", return_value=SubQuestionPlan(sub_questions=["  ", "x"])):
        check("blank sub-questions are dropped", decompose_question("q") == ["x"])

    with patch("agents.critic_agent.invoke_structured", return_value=verdict) as fake_call:
        evaluate_evidence("q", [make_doc(1, paper="Z.pdf")], papers=["Z.pdf", "Y.pdf"])
        sent = fake_call.call_args[0][2]
        check("critic sends the paper list to the LLM", "Y.pdf" in sent["papers"])
        check("critic sees which paper each chunk is from", "Z.pdf" in sent["evidence"])

    try:
        CRITIC_PROMPT.format_messages(question="q", evidence="e", papers="p")
        DECOMPOSE_PROMPT.format_messages(question="q", papers="p", max_sub_questions=3)
        check("critic and planner prompts fill in with no missing variable", True)
    except Exception as error:
        check("critic and planner prompts fill in with no missing variable", False, str(error))

    from core.rag_pipeline import get_llm

    with patch.dict(os.environ, {"GROQ_API_KEY": "dummy-key"}):
        llm = get_llm(streaming=False)
        check("get_llm(streaming=False) is really non-streaming", llm.streaming is False)
        check("get_llm uses max_retries from config", llm.max_retries == config.LLM_MAX_RETRIES)
        check("get_llm uses the model name from config", llm.model_name == config.LLM_MODEL_NAME)
        check("get_llm streams by default (chat UI)", get_llm().streaming is True)


# ===========================================================================
# OFFLINE 5: the agent loop, with fake search / critic / LLM
# ===========================================================================
def _sparse_with_one_paper():
    from core.sparse_store import SparseStore

    store = SparseStore()
    store.build([make_doc(99, paper="A.pdf")])
    return store


@contextmanager
def _patched_workflow(search_fn, verdicts, calls, decompose=None):
    import graph.workflow as wf
    from agents.critic_agent import EvidenceVerdict

    def fake_search(query, vector_store, sparse_store):
        calls["search"] += 1
        return search_fn(query, calls["search"])

    def fake_critic(question, documents, papers=None):
        calls["critic"] += 1
        sufficient = verdicts[min(calls["critic"] - 1, len(verdicts) - 1)]
        return EvidenceVerdict(sufficient=sufficient, reason=f"reason {calls['critic']}")

    def fake_llm(streaming=True):
        calls["llm"] += 1
        return FakeListChatModel(responses=["FAKE ANSWER"])

    patches = [
        patch.object(wf, "rewrite_query", lambda question: question),
        patch.object(wf, "rewrite_query_for_retry", lambda question, previous_query, reason: previous_query + " (retry)"),
        patch.object(wf, "hybrid_search_with_rerank", fake_search),
        patch.object(wf, "evaluate_evidence", fake_critic),
        patch.object(wf, "get_llm", fake_llm),
    ]
    if decompose is not None:
        patches.append(patch.object(wf, "decompose_question", lambda question, papers=None: decompose))

    for p in patches:
        p.start()
    try:
        yield wf
    finally:
        for p in patches:
            p.stop()


def test_agent_loop():
    section("5. Agent loop (fake search, fake critic, fake LLM)")

    # --- 5a. enough evidence at once -> no retry
    calls = {"search": 0, "critic": 0, "llm": 0}
    with _patched_workflow(lambda q, n: [make_doc(1), make_doc(2)], [True], calls) as wf:
        state = wf.run_research("q", object(), _sparse_with_one_paper())
    check("sufficient at once: 0 retries, 1 critic call", state["retry_count"] == 0 and calls["critic"] == 1)
    check("sufficient at once: answer is written", state["final_answer"] == "FAKE ANSWER")

    # --- 5b. insufficient, then NEW evidence, then sufficient
    batches = [[make_doc(1), make_doc(2)], [make_doc(3), make_doc(4)]]
    calls = {"search": 0, "critic": 0, "llm": 0}
    with _patched_workflow(lambda q, n: batches[n - 1], [False, True], calls) as wf:
        state = wf.run_research("q", object(), _sparse_with_one_paper())
    check("retry: 1 retry, 2 critic calls", state["retry_count"] == 1 and calls["critic"] == 2)
    check("retry: evidence is COLLECTED (new first, old kept)", ids(state["evidence"]) == ["c3", "c4", "c1", "c2"], str(ids(state["evidence"])))
    check("retry: the retry query is different from the first", state["attempts"][1]["search_query"] != state["attempts"][0]["search_query"])

    # --- 5c. the retry finds nothing new -> stop early, skip the 2nd critic call
    calls = {"search": 0, "critic": 0, "llm": 0}
    with _patched_workflow(lambda q, n: [make_doc(1), make_doc(2)], [False], calls) as wf:
        state = wf.run_research("q", object(), _sparse_with_one_paper(), max_retries=2)
    check("no new evidence: stops after 1 retry", state["retry_count"] == 1, f"retry_count={state['retry_count']}")
    check("no new evidence: the pointless 2nd critic call is skipped", calls["critic"] == 1, f"critic calls={calls['critic']}")
    check("no new evidence: the stop is written in the attempts log", "no new evidence" in state["attempts"][-1]["reason"].lower())
    check("no new evidence: an answer is still written", state["final_answer"] == "FAKE ANSWER")

    # --- 5d. always insufficient, always new evidence -> stops at max_retries
    calls = {"search": 0, "critic": 0, "llm": 0}
    with _patched_workflow(lambda q, n: [make_doc(10 * n), make_doc(10 * n + 1)], [False], calls) as wf:
        state = wf.run_research("q", object(), _sparse_with_one_paper(), max_retries=2)
    check("retry budget: stops at max_retries (2)", state["retry_count"] == 2 and calls["critic"] == 3)
    check("retry budget: still answers (best effort)", state["final_answer"] == "FAKE ANSWER")
    check("evidence never grows past the cap", len(state["evidence"]) <= config.MAX_EVIDENCE_CHUNKS)

    # --- 5e. nothing found at all -> honest fixed answer, NO LLM call for the answer
    calls = {"search": 0, "critic": 0, "llm": 0}
    with _patched_workflow(lambda q, n: [], [False], calls) as wf:
        state = wf.run_research("q", object(), _sparse_with_one_paper())
        no_evidence_text = wf.NO_EVIDENCE_ANSWER
    check("no evidence: honest fixed answer", state["final_answer"] == no_evidence_text)
    check("no evidence: the LLM is NOT asked to write an answer", calls["llm"] == 0, f"llm calls={calls['llm']}")


def test_multi_question():
    section("6. Multi-question research (fake search, fake critic, fake LLM)")

    good_search = lambda q, n: [make_doc(n, text=f"evidence for {q}")]

    # --- 6a. three sub-questions
    calls = {"search": 0, "critic": 0, "llm": 0}
    with _patched_workflow(good_search, [True], calls, decompose=["s1", "s2", "s3"]) as wf:
        result = wf.run_multi_research("big", object(), _sparse_with_one_paper())
    check("3 sub-questions are all researched", len(result["sub_results"]) == 3 and all(r["error"] is None for r in result["sub_results"]))
    check("3 answers + 1 combining call = 4 LLM calls", calls["llm"] == 4, f"llm calls={calls['llm']}")

    # --- 6b. only one sub-question -> no combining step
    calls = {"search": 0, "critic": 0, "llm": 0}
    with _patched_workflow(good_search, [True], calls, decompose=["only"]) as wf:
        result = wf.run_multi_research("simple", object(), _sparse_with_one_paper())
    check("1 sub-question: combining step is skipped (1 LLM call)", calls["llm"] == 1, f"llm calls={calls['llm']}")
    check("1 sub-question: its answer is the final answer", result["final_answer"] == "FAKE ANSWER")

    # --- 6c. one sub-question crashes, the others continue
    def sometimes_crash(q, n):
        if "BOOM" in q:
            raise RuntimeError("simulated rate limit")
        return [make_doc(n, text=f"evidence for {q}")]

    calls = {"search": 0, "critic": 0, "llm": 0}
    with _patched_workflow(sometimes_crash, [True], calls, decompose=["ok1", "BOOM", "ok2"]) as wf:
        result = wf.run_multi_research("big", object(), _sparse_with_one_paper())
    errors = [r["error"] is not None for r in result["sub_results"]]
    check("one failing sub-question does not stop the others", errors == [False, True, False], str(errors))
    check("the final answer is still produced", result["final_answer"] == "FAKE ANSWER")

    # --- 6d. everything crashes -> a clear error, not a fake answer
    def always_crash(q, n):
        raise RuntimeError("simulated outage")

    calls = {"search": 0, "critic": 0, "llm": 0}
    with _patched_workflow(always_crash, [True], calls, decompose=["a", "b"]) as wf:
        try:
            wf.run_multi_research("big", object(), _sparse_with_one_paper())
            check("all sub-questions failing raises a clear error", False)
        except RuntimeError as error:
            check("all sub-questions failing raises a clear error", "Every sub-question failed" in str(error))


# ===========================================================================
# LIVE: the real pipeline
# ===========================================================================
PAPERS = [
    "data/sample_research_paper.pdf",
    "data/sample_paper_2_denseboost.pdf",
    "data/sample_paper_3_lexirank.pdf",
]


def run_live():
    from dotenv import load_dotenv

    load_dotenv()

    from core.document_utils import SmartPDFProcessor
    from core.sparse_store import SparseStore
    from core.vector_store import get_embedding_model, get_vector_store, ingest_documents
    from graph.workflow import run_multi_research

    section("LIVE: real AstraDB + real Groq (answers vary a little between runs)")
    print("If AstraDB was asleep you may see a 503: wake it in the Astra dashboard and run again.\n")

    missing = [p for p in PAPERS if not os.path.exists(p)]
    if missing:
        check("sample papers exist in data/", False, f"missing: {missing}")
        return

    embeddings = get_embedding_model()
    vector_store = get_vector_store(embeddings)
    vector_store.clear()

    processor = SmartPDFProcessor(embeddings=embeddings)
    all_chunks = []
    for path in PAPERS:
        chunks = processor.process_pdf(path)
        print(f"  {path}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    check("every chunk is within the max size", max(len(c.page_content) for c in all_chunks) <= config.SEMANTIC_MAX_CHUNK_CHARS)
    check("chunk ids are unique", len({c.metadata["chunk_id"] for c in all_chunks}) == len(all_chunks))

    ingest_documents(vector_store, all_chunks)
    sparse_store = SparseStore()
    sparse_store.build(all_chunks)
    check("the sparse index knows all 3 papers", len(sparse_store.paper_ids()) == 3, str(sparse_store.paper_ids()))

    def show(result):
        print("\n  sub-questions:")
        for r in result["sub_results"]:
            print(f"   - {r['sub_question']}   (retries: {r['retries_used']}, error: {r['error']})")
            for attempt in r["attempts"]:
                print(f"       query: {attempt['search_query']!r} -> sufficient={attempt['sufficient']}")
        print(f"\n  FINAL ANSWER:\n{result['final_answer']}\n")

    # --- Q1: simple question
    print("\n--- Q1: simple question ---")
    r1 = run_multi_research("What Recall at 5 did DenseBoost achieve?", vector_store, sparse_store)
    show(r1)
    check("Q1: stays ONE sub-question", len(r1["sub_questions"]) == 1, str(r1["sub_questions"]))
    check("Q1: the answer contains 0.68", "0.68" in r1["final_answer"])

    # --- Q2: comparison across papers
    print("\n--- Q2: comparison across the three papers ---")
    r2 = run_multi_research(
        "Compare the retrieval techniques used in these papers and tell me which techniques improve retrieval.",
        vector_store,
        sparse_store,
    )
    show(r2)
    check("Q2: split into 2 or more sub-questions", 2 <= len(r2["sub_questions"]) <= config.MAX_SUB_QUESTIONS, str(r2["sub_questions"]))
    check(
        "Q2: no sub-question asks the user for clarification",
        not any(("referring to" in q.lower() or "which papers" in q.lower()) for q in r2["sub_questions"]),
    )
    answer2 = r2["final_answer"].lower()
    check("Q2: the answer mentions all three papers", all(name in answer2 for name in ("ragfusion", "denseboost", "lexirank")))

    # --- Q3: a question the papers do not answer
    print("\n--- Q3: question that is NOT in the papers ---")
    r3 = run_multi_research("What GPU hardware was used to run the experiments?", vector_store, sparse_store)
    show(r3)
    check(
        "Q3: retries stay within the budget",
        all(r["retries_used"] <= config.MAX_RETRIES for r in r3["sub_results"]),
    )
    check("Q3: the answer admits the information is missing", any(w in r3["final_answer"].lower() for w in ("not", "no ", "cannot", "n't")))


# ===========================================================================
def main():
    only_offline = len(sys.argv) > 1 and sys.argv[1].lower() == "offline"

    test_sparse_store()
    test_chunker()
    test_document_utils()
    test_fusion_and_merge()
    test_hybrid_fallback()
    test_agents()
    test_agent_loop()
    test_multi_question()

    if not only_offline:
        try:
            run_live()
        except Exception as error:
            check("live run finished without an exception", False, f"{type(error).__name__}: {error}")

    passed, total = sum(RESULTS), len(RESULTS)
    print("\n" + "=" * 70)
    print(f"RESULT: {passed}/{total} checks passed")
    print("=" * 70)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()