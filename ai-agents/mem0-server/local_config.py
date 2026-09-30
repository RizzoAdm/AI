"""Config local do Mem0 server: Ollama + Qdrant, sem OpenAI/pgvector.
Importada pelo main.py oficial via patch_main.py (aplicado no build)."""
import os

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("MEM0_LLM_MODEL", "gpt-oss:20b-64k")
EMBED_MODEL = os.environ.get("MEM0_EMBED_MODEL", "nomic-embed-text:latest")
EMBED_DIMS = int(os.environ.get("MEM0_EMBED_DIMS", "768"))
QDRANT_HOST = os.environ.get("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION = os.environ.get("MEM0_COLLECTION", "mem0_memories")
HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "/app/history/history.db")

LOCAL_CONFIG = {
    "version": "v1.1",
    "llm": {
        "provider": "ollama",
        "config": {
            "model": LLM_MODEL,
            "temperature": 0.1,
            "max_tokens": int(os.environ.get("MEM0_LLM_MAX_TOKENS", "4096")),
            "ollama_base_url": OLLAMA_BASE_URL,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": EMBED_MODEL,
            "embedding_dims": EMBED_DIMS,
            "ollama_base_url": OLLAMA_BASE_URL,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QDRANT_HOST,
            "port": QDRANT_PORT,
            "collection_name": COLLECTION,
            "embedding_model_dims": EMBED_DIMS,
        },
    },
    "history_db_path": HISTORY_DB_PATH,
}

# Agentes cuja sincronizacao automatica (infer=True) e descartada pelo servidor.
# Gravacoes explicitas (infer=False, ex.: mem0_add) continuam normais.
SKIP_INFER_AGENTS = {a.strip() for a in os.environ.get("MEM0_SKIP_INFER_AGENTS", "").split(",") if a.strip()}
