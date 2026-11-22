# InsightRAG — Multi-Document Contract, Policy & Compliance Auditing Assistant

An end-to-end, context-grounded multi-document RAG pipeline built for **enterprise vendor contract and compliance auditing**. Upload internal security policies, regulatory frameworks (e.g., NIST CSF 2.0), and vendor agreements (e.g., AWS & Stripe DPAs) to run cross-document gap analyses with verified page-level citations and strict anti-hallucination refusals.

---

## Why This Exists

Before onboarding third-party vendors, security and compliance teams manually cross-check dozens of pages of Data Processing Agreements (DPAs) against internal security policies and frameworks like NIST CSF 2.0. Naive single-vector RAG pipelines break on this workload due to:
- **Cross-document pollution** (chunks from Vendor A leaking into Vendor B queries)
- **Exact-term misses** (dense embeddings missing specific clause IDs like `GV.SC-05` or legal terms)
- **Follow-up breakdown** (multi-turn comparisons losing earlier context or triggering redundant vector searches)
- **Silent citation hallucinations** (confusing internal section numbers like `Section 14` with PDF page numbers)

InsightRAG solves these failure modes using a **1-call Pydantic query router**, **hybrid dense + BM25 search with RRF fusion**, **Cross-Encoder re-ranking**, and **metadata-locked citations**.

---

## System Architecture

```text
User Question + Last 10 Chat Messages (text only, raw chunks stripped)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  1-Call Agentic Query Planner (Pydantic Structured Output)│
│  • needs_retrieval (bool)                                 │
│  • direct_answer (fast-path bypass for meta/history Qs)   │
│  • search_query (keyword + synonym-expanded query)        │
│  • standalone_question (full resolved user intent)        │
│  • target_documents (exact PDF routing filter)            │
└──────────────┬───────────────────────────┬────────────────┘
               │ needs_retrieval=False     │ needs_retrieval=True
               ▼                           ▼
   Direct History Response      Per-Document Filtered Hybrid Search
      (0 Qdrant calls)          (Qdrant Cloud + FastEmbed: Dense + BM25)
                                           │
                                           ▼
                                Reciprocal Rank Fusion (RRF, Top 15/doc)
                                           │
                                           ▼
                                Cross-Encoder Re-Ranking
                                (Top 5 single-doc | Top 4/doc multi-doc)
                                           │
                                           ▼
                                Grounded Answer Generation (Groq LLM)
                                • Strict context-only rules & exact refusals
                                • Metadata-only page citations [File, Page X]
```

---

## Key Engineering Features

- **1-Call Structured Query Router (`QueryPlan`)**: Resolves pronouns across multi-turn conversations (`messages[-10:]`), routes queries strictly to relevant PDFs via `doc_hash` metadata filters, and separates `search_query` (concise retrieval keywords) from `standalone_question` (full comparison/analysis instructions).
- **Zero-Retrieval Fast Paths**:
  - Skips the router LLM call entirely on first-turn single-document queries.
  - Bypasses Qdrant and re-ranking completely (`needs_retrieval=False`) for greetings, meta-questions, and summaries of prior turns.
- **Native Hybrid Search + Cross-Encoder Re-Ranking**: Combines semantic vectors (`all-MiniLM-L6-v2`) with sparse lexical matching (`Qdrant/bm25` with IDF) fused via RRF in Qdrant Cloud, followed by Cross-Encoder scoring (`ms-marco-MiniLM-L-6-v2`).
- **SHA-256 Deduplication & Self-Healing Vector Store**: Hashes uploaded PDFs to skip redundant extraction/indexing across sessions and automatically recreates the Qdrant collection and `doc_hash` payload index if missing.
- **Anti-Hallucination & Citation Transparency**: Enforces exact refusals (`Not found in document.`) for out-of-scope queries, defends against prompt injection inside PDFs, and exposes retrieved context and re-rank scores in the Streamlit UI.

---

## Tech Stack

| Layer | Implementation |
| :--- | :--- |
| **LLM Inference** | Groq (`openai/gpt-oss-120b`, `temperature=0`) via `langchain-groq` |
| **Structured Routing** | Pydantic v2 (`llm.with_structured_output(QueryPlan)`) |
| **Vector Store** | Qdrant Cloud (`insight_documents` collection with `doc_hash` keyword index) |
| **Hybrid Embeddings** | FastEmbed native Qdrant `models.Document` — `sentence-transformers/all-MiniLM-L6-v2` (384-d Cosine) + `Qdrant/bm25` (IDF) |
| **Re-Ranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Ingestion & Chunking** | `pypdf` (in-memory stream, `strict=False`) + `RecursiveCharacterTextSplitter` (`chunk_size=900`, `chunk_overlap=150`) |
| **Frontend** | Streamlit persistent chat UI with expandable retrieved context and scored source panels |

---

## Benchmark Corpus & Error Analysis

Evaluated across **4 real-world enterprise PDFs (60 pages, 207 indexed chunks)**:
- `AWS_GDPR_DPA.pdf` — 10 pages / 42 chunks (Vendor Agreement #1)
- `Stripe_Global_DPA.pdf` — 15 pages / 56 chunks (Vendor Agreement #2)
- `NIST_CSF_2.0.pdf` — 32 pages / 103 chunks (Cybersecurity Compliance Framework)
- `Internal_Third_Party_Security_Policy.pdf` — 3 pages / 6 chunks (Internal Company Policy)

### Production Failure Modes Caught & Resolved

| Failure Mode Observed | Root Cause | Engineering Fix | Verified Outcome |
| :--- | :--- | :--- | :--- |
| **Dropped comparisons & skipped calculations** | Overloaded `plan.search_query` stripped action verbs before reaching `generate_answer()`. | Decoupled `search_query` (retrieval) from `standalone_question` (generation) in `QueryPlan`. | Multi-doc gap tables and comparative follow-ups now preserve full user intent. |
| **Context leakage & forgotten early turns** | Passing raw PDF chunks in history bloated router prompts; 4-message window dropped earlier turns. | Stripped raw chunks from router history and expanded window to `messages[-10:]`. | Clean pronoun/topic resolution across multi-turn audits with lower router latency. |
| **Section-number vs. page-number citation hallucinations** | LLM cited clause headers (e.g., `Section 14` → `Page 14`, `Section 9` → `Page 9`) instead of PDF pages. | Enforced metadata-only citation rule tied strictly to `[Document: ... \| Page: X]` headers. | Eliminated page citation errors across multi-document NIST/AWS/Stripe audits (e.g., AWS fixed to exact `Page 4` and `Page 6`). |
| **Missed bottom-of-page & secondary clauses** | Fixed `top_k=3` per document crowded out lower-ranked relevant chunks on dense pages. | Dynamic re-ranker cutoff (`per_doc_k = 5` single-doc, `4` multi-doc) + domain synonym guidance. | Recovered sub-processor flow-down clauses and deep-page obligations cleanly. |
| **Cached Qdrant 404 on collection reset** | `@st.cache_resource` reused a cached client without re-initializing a recreated collection. | Added `client.collection_exists()` guard inside `index_chunks()` before upsert. | Zero-downtime automatic collection and payload-index recreation. |

---

## Project Structure

```text
├── app.py                              # Streamlit chat UI, deduplicated upload & RAG orchestration
├── src/
│   ├── config/
│   │   └── settings.py                 # Model configs, chunking, retrieval K, and demo prompts
│   ├── ingestion/
│   │   └── pdf_processor.py            # Thread-safe in-memory PDF extraction & chunking
│   ├── retrieval/
│   │   ├── qdrant_manager.py           # Qdrant Cloud client, FastEmbed hybrid search & RRF fusion
│   │   └── reranker.py                 # Cross-Encoder candidate re-ranking
│   ├── generation/
│   │   └── answer.py                   # 1-call Pydantic QueryPlan router & grounded answer generator
│   └── utils/
│       └── secrets.py                  # Credential loader for Streamlit secrets / env vars
└── requirements.txt
```

---

## Local Setup

1. **Clone and install dependencies**
   ```bash
   git clone [https://github.com/Salaam7867/rag-pdf-chat.git](https://github.com/Salaam7867/rag-pdf-chat.git)
   cd rag-pdf-chat
   pip install -r requirements.txt
   ```

2. **Configure secrets**  
   Create `.streamlit/secrets.toml` (or export as environment variables):
   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   QDRANT_URL = "[https://your-cluster-url.qdrant.io](https://your-cluster-url.qdrant.io)"
   QDRANT_API_KEY = "your_qdrant_api_key"
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```
