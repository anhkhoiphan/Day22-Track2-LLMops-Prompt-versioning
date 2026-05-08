"""
Step 3 — RAGAS Evaluation
===========================
Runs all 50 QA pairs through both prompt versions, evaluates with 4 RAGAS
metrics (faithfulness, answer_relevancy, context_recall, context_precision),
prints a V1 vs V2 comparison table, and saves data/ragas_report.json.

Deliverable: faithfulness ≥ 0.8 for at least one prompt version.
⏰ NOTE: Takes ~15-20 minutes. Start early!
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np

import config  # sets LangSmith env vars
from qa_pairs import QA_PAIRS

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

# ── Prompt templates ─────────────────────────────────────────────────────────
SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── Build FAISS vector store ─────────────────────────────────────────────────
def build_vectorstore():
    text     = config.KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks   = splitter.split_text(text)
    return FAISS.from_texts(chunks, config.get_embeddings())


# ── Single RAG call (returns answer + list of context strings) ────────────────
@traceable(name="ragas-rag", tags=["ragas", "step3"])
def run_rag(retriever, llm, prompt, question: str) -> dict:
    docs     = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]   # list[str] — RAGAS needs this
    ctx_str  = "\n\n".join(contexts)
    answer   = (prompt | llm | StrOutputParser()).invoke(
        {"context": ctx_str, "question": question}
    )
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = config.get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\nRunning 50 questions with prompt {prompt_version} ...")
    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


# ── Build RAGAS EvaluationDataset ─────────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


# ── Run RAGAS evaluation ──────────────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    print(f"\n📐 Running RAGAS evaluation for prompt {version} ...")

    dataset  = build_ragas_dataset(rag_results)
    llm_eval = config.get_llm(temperature=0.0)
    emb_eval = config.get_embeddings()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
    )

    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        scores[key] = float(np.mean([v for v in raw if v is not None]))

    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    vectorstore = build_vectorstore()

    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    print("\n" + "=" * 60)
    print(f"  {'Metric':<30} {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 60)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "← V1" if s1 >= s2 else "← V2"
        print(f"  {metric:<30} {s1:>8.4f}  {s2:>8.4f}  {winner}")
    print("=" * 60)

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"✅ Target met: faithfulness = {best_faith:.4f}")
    else:
        print(f"⚠️  Below target ({best_faith:.4f}). Try adjusting chunking or prompts.")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": bool(best_faith >= 0.8),
    }
    config.RAGAS_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"💾 Saved {config.RAGAS_REPORT_PATH}")


if __name__ == "__main__":
    main()
