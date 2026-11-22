"""Central configuration for the InsightRAG pipeline."""

APP_TITLE = "InsightRAG"
APP_CAPTION = "Multi-Document Contract, Policy & Compliance Auditing Assistant (Hybrid RAG + Agentic Routing)"

# Chunking settings (enlarged to preserve clause and paragraph context)
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Retrieval & Re-ranking counts
CANDIDATE_POOL_K = 15  # Candidates retrieved from Hybrid Search (Dense + Sparse)
TOP_K_FINAL = 4        # Top chunks retained after Cross-Encoder re-ranking

# Models
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL_NAME = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0

# Qdrant configuration
QDRANT_COLLECTION_NAME = "insight_documents"
DENSE_VECTOR_SIZE = 384  # Dimension for all-MiniLM-L6-v2

SUGGESTED_QUESTIONS = [
    "Compare data breach and security incident notification obligations across the uploaded policies and vendor agreements. Highlight any gaps.",
    "Compare post-termination data return and deletion obligations across the vendor agreements.",
    "Which vendor imposes stricter restrictions on customer audit rights, and what are the exact limits?",
    "How do the vendors' security controls align with the governance and protection goals in the compliance framework?",
    "What is the guaranteed monthly uptime SLA percentage and service credit refund schedule?",
]