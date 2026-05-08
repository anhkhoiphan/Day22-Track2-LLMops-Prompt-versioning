"""
Step 2 — Prompt Hub & A/B Routing
===================================
Pushes two prompt versions to LangSmith Prompt Hub, pulls them back,
and deterministically routes each of 50 questions to V1 or V2 using MD5.

Deliverable: 2 named prompts in Prompt Hub + ≥ 50 more LangSmith traces.
"""

import hashlib

import config  # sets LangSmith env vars
from qa_pairs import SAMPLE_QUESTIONS

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

# ── Two distinct system prompts ──────────────────────────────────────────────
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

PROMPT_V1_NAME = "rag-prompt-v1"
PROMPT_V2_NAME = "rag-prompt-v2"


# ── Push prompts to Prompt Hub ───────────────────────────────────────────────
def push_prompts_to_hub(client: Client):
    for name, template, desc in [
        (PROMPT_V1_NAME, PROMPT_V1, "V1 – concise 2-4 sentence answers"),
        (PROMPT_V2_NAME, PROMPT_V2, "V2 – structured 3-5 sentence answers"),
    ]:
        try:
            url = client.push_prompt(name, object=template, description=desc)
            print(f"✅ Pushed '{name}' → {url}")
        except Exception as e:
            print(f"⚠️  Could not push '{name}': {e}")


# ── Pull prompts from Prompt Hub ─────────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    prompts = {}
    for name, fallback in [(PROMPT_V1_NAME, PROMPT_V1), (PROMPT_V2_NAME, PROMPT_V2)]:
        try:
            prompts[name] = client.pull_prompt(name)
            print(f"↓ Pulled '{name}' from Hub")
        except Exception:
            prompts[name] = fallback
            print(f"ℹ️  Using local fallback for '{name}'")
    return prompts


# ── Deterministic A/B routing ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── Build FAISS vector store ─────────────────────────────────────────────────
def build_vectorstore():
    text     = config.KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks   = splitter.split_text(text)
    return FAISS.from_texts(chunks, config.get_embeddings())


# ── Traced A/B query ─────────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    docs    = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer  = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    return {"question": question, "answer": answer, "version": version}


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    client = Client(api_key=config.LANGSMITH_API_KEY)

    push_prompts_to_hub(client)
    prompts   = pull_prompts_from_hub(client)
    vectorstore = build_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = config.get_llm()

    v1_count = v2_count = 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        result = ask_ab(retriever, llm, prompt, question, version_tag)
        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}...")

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1

    print(f"\n📊 Routing summary: V1={v1_count} queries | V2={v2_count} queries")
    print(f"✅ {len(SAMPLE_QUESTIONS)} traces sent to LangSmith project '{config.LANGSMITH_PROJECT}'")


if __name__ == "__main__":
    main()
