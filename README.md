# StatefulRAG
**Stateful Multi-User RAG with Memory, Caching & pgvector**

Multi-turn RAG system where each user has isolated conversation memory, repeated questions are served from a semantic cache, and every LLM call is traced in Langfuse. Built on a LangGraph agent with PostgreSQL + pgvector for persistence.

**Intent:** To move beyond "toy" RAG scripts and tackle the actual engineering hurdles of production-grade AI. This project is a deep dive into building a sophisticated, multi-tenant system that prioritizes state management (multi-user memory), performance optimization (semantic caching), and observability (Langfuse tracing). Essentially, it's about shifting from simple "input-output" loops to a robust, agentic architecture that can handle the complexities of real-world usage.

>**Tip:** You can refer the notebook for step by step understanding of the repo.

## What This Is

- **Stateful agent** — LangGraph replaces the linear LangChain chain
- **Per-user memory** — conversation history stored in PostgreSQL per user+session
- **History compression** — older messages summarized by the LLM when history grows long
- **Semantic cache** — Redis stores question+answer pairs, similar future questions skip the LLM entirely
- **pgvector** — replaces ChromaDB with a production-grade vector store inside PostgreSQL
- **Full observability** — every graph node traced in Langfuse Cloud



## Stack

| Component | Tool |
|---|---|
| LLM | `gemma4:e2b` via Ollama (local) |
| Embeddings | `nomic-embed-text` via Ollama (local, 768-dim) |
| Vector Store | pgvector (PostgreSQL) |
| Agent Orchestration | LangGraph |
| Memory | PostgreSQL (`chat_messages`, `chat_summaries`) |
| Semantic Cache | Redis |
| Observability | Langfuse Cloud v4 |
| UI | Streamlit |
| Package Manager | `uv` |
| Infrastructure | Docker (PostgreSQL + pgvector + Redis) |



## Project Structure

```
stateful_rag_l2/
├── .env                        # secrets and config (never commit)
├── .env.example
├── docker-compose.yml          # PostgreSQL + pgvector + Redis
├── pyproject.toml
├── uv.lock
│
├── data/
│   └── papers/                 # put PDFs here
│
├── notebooks/
│   └── stateful_agent.ipynb   # full pipeline walkthrough
│
├── stateful_rag/
│   ├── config.py               # single source of truth for all settings
│   ├── ingestion/
│   │   ├── loader.py           # PDF → markdown (pymupdf4llm)
│   │   └── chunker.py          # recursive markdown-aware chunking
│   ├── embeddings/
│   │   └── embedder.py         # nomic-embed-text, embed_texts + embed_query
│   ├── llm/
│   │   └── provider.py         # ChatOllama, gemma4:e2b
│   ├── vectorstore/
│   │   └── pgvector_store.py   # add_chunks, similarity_search, stats
│   ├── memory/
│   │   ├── pg_memory.py        # read/write chat history to PostgreSQL
│   │   └── summarizer.py       # compress old messages with gemma4
│   ├── cache/
│   │   └── redis_cache.py      # semantic cache: embed, store, lookup
│   ├── agent/
│   │   ├── state.py            # AgentState TypedDict
│   │   ├── nodes.py            # one function per graph node
│   │   └── graph.py            # LangGraph graph definition + ask()
│   └── observability/
│       └── langfuse_client.py  # Langfuse v4 setup + node tracing
│
├── ui/
│   └── app.py                  # Streamlit multi-user chat interface
│
└── tests/
```



## Prerequisites

| Tool | Purpose | Install |
|---|---|---|
| Python 3.11 | Runtime | via `uv` |
| `uv` | Package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Ollama | Local LLM + embeddings | [ollama.com](https://ollama.com) |
| Docker Desktop | PostgreSQL + Redis | [docker.com](https://docker.com) |
| Langfuse account | Observability | [cloud.langfuse.com](https://cloud.langfuse.com) (free) |

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd stateful_rag_l2

uv python pin 3.11
uv sync
source .venv/bin/activate
```

### 2. Pull Ollama models

```bash
ollama pull gemma4:e2b
ollama pull nomic-embed-text
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:e2b

# Embeddings
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=docmind
POSTGRES_USER=docmind
POSTGRES_PASSWORD=docmind

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_TTL_SECONDS=3600
REDIS_SIMILARITY_THRESHOLD=0.15

# Memory
MEMORY_LAST_N_MESSAGES=6
MEMORY_SUMMARIZE_AFTER=10

# Langfuse Cloud
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```
You can use the db credentials to connect to an SQL client, I was using DBeaver.

### 4. Start infrastructure

```bash
docker compose up -d
docker compose ps  # both should show (healthy)
```

### 5. Create database schema

```bash
docker exec -i <container_name> psql -U <user_name> -d <user_name> << 'EOF'

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    chunk_id    TEXT UNIQUE NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB,
    embedding   vector(768)
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          SERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_summaries (
    id            SERIAL PRIMARY KEY,
    user_id       TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    summary       TEXT NOT NULL,
    message_count INT NOT NULL,
    created_at    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_user_session
    ON chat_messages(user_id, session_id);

CREATE INDEX IF NOT EXISTS idx_chat_summaries_user_session
    ON chat_summaries(user_id, session_id);

CREATE INDEX IF NOT EXISTS idx_documents_chunk_id
    ON documents(chunk_id);

EOF
```

### 6. Ingest a paper

Upload via Streamlit UI or run the notebook top to bottom.



## Run

### Streamlit UI

```bash
streamlit run ui/app.py
```

Opens at `http://localhost:8501`

1. Select a user (Dave, Mike, or custom) in the sidebar
2. Upload a PDF → click Ingest Paper
3. Ask questions — each user has isolated memory
4. Switch users — notice different conversation history

### Redis Commander (cache browser)

Add to `docker-compose.yml` under services:

```yaml
  redis-commander:
    image: rediscommander/redis-commander:latest
    container_name: stateful_rag_redis_ui
    environment:
      REDIS_HOSTS: local:redis:6379
    ports:
      - "8081:8081"
    depends_on:
      - redis
```

Open `http://localhost:8081` to browse cache entries and TTLs.


## How It Works

### Agent Graph

```
Question + user_id + session_id
    │
    ▼
load_memory        fetch summary + recent messages from PostgreSQL
    │
    ▼
check_cache        embed question → cosine search in Redis
    ├── HIT   ──────────────────────────────────────────┐
    │                                                   │
    ▼                                                   │
retrieve           embed query → pgvector similarity search
    │                                                   │
    ▼                                                   │
generate           RAG prompt + memory context → gemma4 │
    │                                                   │
    ▼                                                   │
save_memory   ◄─────────────────────────────────────────┘
    │          save human + assistant messages to PostgreSQL
    │          trigger summarization if count > threshold
    ▼
cache_response     store question+answer in Redis (on miss only)
    │
    ▼
  answer + sources + cache_hit flag
```

### Memory Strategy

**Last-N messages:** Always fetch the 6 most recent message pairs from PostgreSQL for the current user+session. Injected into the prompt as recent conversation history.

**Summarization:** When message count exceeds 10 pairs (20 messages), gemma4 compresses the oldest 20 messages into a summary paragraph stored in `chat_summaries`. Those messages are deleted. The summary is prepended to future prompts as older context.

**Result:** Prompt always contains summary (older context) + recent messages (precise context). History never grows unbounded.

### Semantic Cache

1. Incoming question is embedded with `nomic-embed-text`
2. Cosine distance computed against all cached question vectors in Redis
3. Distance < 0.15 → cache hit → return stored answer instantly (~0.1s)
4. Distance ≥ 0.15 → cache miss → full pipeline → store result in Redis with 1hr TTL

"How does multi-head attention work?" and "Explain multi-head attention" both hit the same cache entry.

### User Isolation

Each user has their own rows in `chat_messages` and `chat_summaries` keyed by `user_id + session_id`. Dave's conversation history is completely invisible to Mike. The semantic cache is shared — both users benefit from cached answers.


## Database Schema

```sql
-- Vector store
documents (id, chunk_id, content, metadata JSONB, embedding vector(768))

-- Per-user chat history
chat_messages (id, user_id, session_id, role, content, created_at)

-- Compressed history
chat_summaries (id, user_id, session_id, summary, message_count, created_at)
```


## Observability

Every agent run creates a Langfuse trace with spans per node:

```
agent_run
├── load_memory      input: user_id | output: has_summary, message_count
├── check_cache      input: question | output: cache_hit
├── retrieve         input: question | output: chunk_ids, similarities
├── rag_generate     full LLM trace: prompt + response + token counts
├── save_memory      output: messages_saved, summarization_triggered
└── cache_response   output: stored_in_cache
```

View at [cloud.langfuse.com](https://cloud.langfuse.com) → Traces.


## Key Design Decisions

**Why LangGraph over LangChain chain?**
Linear chains can't branch. The cache hit path skips retrieval and generation entirely — this requires conditional routing which LangGraph handles natively via `add_conditional_edges`.

**Why pgvector over ChromaDB?**
Single database for vectors, chat history, and summaries. Cosine similarity via `<=>` operator. Production-grade persistence via Docker volumes. HNSW indexing available when scale requires it.

**Why cosine distance threshold 0.15?**
Tested against `nomic-embed-text` on natural language question paraphrases. Below 0.05 is too strict (only catches near-identical phrasing). Above 0.20 risks false cache hits. 0.15 balances recall and precision for research paper Q&A.

**Why summarize at 10 message pairs?**
gemma4's context window is 128K but injecting 20+ full messages into every prompt is wasteful and slows inference. 10 pairs gives enough context before compression. The summary preserves key facts while keeping prompt size bounded.

**Why share cache across users?**
Research paper facts don't change per user. Dave and Mike asking the same question about the Attention paper should get the same answer. Sharing the cache maximises hit rate. Memory (which is personal) remains fully isolated.