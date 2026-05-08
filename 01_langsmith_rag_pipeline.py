"""
Step 1 — LangSmith-instrumented RAG Pipeline
=============================================
Builds a FAISS-backed RAG chain and runs 50 sample questions through it,
each decorated with @traceable so every call appears in LangSmith.

Deliverable: ≥ 50 traces visible in https://smith.langchain.com
"""

import config  # sets LangSmith env vars before any LangChain import
from qa_pairs import SAMPLE_QUESTIONS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

# ── LLM and Embeddings ───────────────────────────────────────────────────────
llm        = config.get_llm()
embeddings = config.get_embeddings()


# ── Build FAISS vector store ─────────────────────────────────────────────────
def build_vectorstore():
    text = config.KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks   = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


# ── RAG prompt template ──────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Use the context below to answer the question.\n"
     "Answer ONLY from the provided context. If the context does not contain the "
     "answer, say: 'I don't have enough information.'\n\nContext:\n{context}"),
    ("human", "{question}"),
])


# ── Build the RAG chain ──────────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ── Traced query function ────────────────────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    return chain.invoke(question)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    vectorstore      = build_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
        print(f"       A: {answer[:120]}\n")

    print(f"✅ {len(SAMPLE_QUESTIONS)} traces sent to LangSmith project '{config.LANGSMITH_PROJECT}'")
    print("   Open https://smith.langchain.com to view traces.")


if __name__ == "__main__":
    main()
