"""
Shared configuration helpers — loaded by all four lab scripts.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root
load_dotenv(Path(__file__).parent / ".env")

# ── LangSmith tracing (must be set BEFORE importing LangChain) ──────────────
os.environ["LANGCHAIN_TRACING_V2"]  = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGCHAIN_PROJECT", "day22-langsmith-lab")
os.environ["LANGCHAIN_ENDPOINT"]    = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# ── OpenAI settings ──────────────────────────────────────────────────────────
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL      = os.getenv("OPENAI_BASE_URL", None)
OPENAI_MODEL         = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

LANGSMITH_API_KEY    = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT    = os.getenv("LANGCHAIN_PROJECT", "day22-langsmith-lab")

DATA_DIR             = Path(__file__).parent / "data"
KNOWLEDGE_BASE_PATH  = DATA_DIR / "knowledge_base.txt"
RAGAS_REPORT_PATH    = DATA_DIR / "ragas_report.json"


def get_llm(temperature: float = 0.0):
    from langchain_openai import ChatOpenAI
    kwargs = dict(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=temperature)
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return ChatOpenAI(**kwargs)


def get_embeddings():
    from langchain_openai import OpenAIEmbeddings
    kwargs = dict(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return OpenAIEmbeddings(**kwargs)
