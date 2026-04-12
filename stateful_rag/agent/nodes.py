import logging
import uuid
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from stateful_rag.agent.state import AgentState
from stateful_rag.config import get_settings
from stateful_rag.embeddings.embedder import embed_query
from stateful_rag.vectorstore.pgvector_store import similarity_search
from stateful_rag.memory.pg_memory import (
    save_message,
    should_summarize,
    get_recent_messages,
    get_latest_summary,
)
from stateful_rag.memory.summarizer import summarize_and_compress
from stateful_rag.cache.redis_cache import cache_lookup, cache_store
from stateful_rag.llm.provider import get_llm
from stateful_rag.observability.langfuse_client import (
    get_langfuse_handler,
    trace_node,
)

logger = logging.getLogger(__name__)

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert research assistant. Answer the question using ONLY the context provided.
If the context does not contain enough information, say "I don't have enough context to answer this."
Do not use prior knowledge. Always cite the chunk ID that supports your answer.

{summary_section}

{history_section}

Retrieved Context:
{context}

Question: {question}

Answer:
""")


def _format_summary_section(summary: str) -> str:
    if not summary:
        return ""
    return f"Conversation Summary (older context):\n{summary}"


def _format_history_section(recent_messages: list[dict]) -> str:
    if not recent_messages:
        return ""
    lines = []
    for m in recent_messages:
        role = "User" if m["role"] == "human" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "Recent Conversation:\n" + "\n".join(lines)


def load_memory(state: AgentState) -> AgentState:
    """Node 1: Load user memory from PostgreSQL."""
    user_id = state["user_id"]
    session_id = state["session_id"]

    # Generate a fresh trace_id for this entire agent run
    # trace_id = str(uuid.uuid4())
    trace_id = uuid.uuid4().hex # For LangFuse compatibility

    summary = get_latest_summary(user_id, session_id) or ""
    recent = get_recent_messages(user_id, session_id)

    trace_node(
        trace_id=trace_id,
        node_name="load_memory",
        user_id=user_id,
        session_id=session_id,
        input_data={"user_id": user_id, "session_id": session_id},
        output_data={
            "has_summary": bool(summary),
            "recent_message_count": len(recent),
        },
    )

    return {
        **state,
        "trace_id": trace_id,
        "summary": summary,
        "recent_messages": recent,
    }


def check_cache(state: AgentState) -> AgentState:
    """Node 2: Check Redis semantic cache."""
    question = state["question"]
    question_embedding = embed_query(question)
    result = cache_lookup(question, question_embedding=question_embedding)

    cache_hit = bool(result)

    trace_node(
        trace_id=state["trace_id"],
        node_name="check_cache",
        user_id=state["user_id"],
        session_id=state["session_id"],
        input_data={"question": question},
        output_data={"cache_hit": cache_hit},
    )

    if result:
        return {
            **state,
            "cache_hit": True,
            "answer": result["answer"],
            "sources": result["sources"],
            "context": "",
        }

    return {
        **state,
        "cache_hit": False,
        "answer": "",
        "sources": [],
        "context": "",
    }


def retrieve(state: AgentState) -> AgentState:
    """Node 3: Retrieve relevant chunks from pgvector."""
    question = state["question"]
    query_embedding = embed_query(question)
    results = similarity_search(query_embedding, k=4)

    context_parts = []
    for r in results:
        context_parts.append(f"[{r['chunk_id']}]\n{r['content']}")
    context = "\n\n---\n\n".join(context_parts)

    sources = [
        {
            "chunk_id": r["chunk_id"],
            "source": r["metadata"].get("source", ""),
            "preview": r["content"][:200],
            "similarity": round(r["similarity"], 4),
        }
        for r in results
    ]

    trace_node(
        trace_id=state["trace_id"],
        node_name="retrieve",
        user_id=state["user_id"],
        session_id=state["session_id"],
        input_data={"question": question},
        output_data={
            "chunks_retrieved": len(results),
            "top_similarity": sources[0]["similarity"] if sources else 0,
            "chunk_ids": [s["chunk_id"] for s in sources],
        },
    )

    return {
        **state,
        "context": context,
        "sources": sources,
    }


def generate(state: AgentState) -> AgentState:
    """Node 4: Generate answer using LLM."""
    llm = get_llm()

    # LangChain handler traces the LLM call itself
    handler = get_langfuse_handler(
        session_id=state["session_id"],
        user_id=state["user_id"],
        trace_name="rag_generate",
    )

    prompt_input = {
        "summary_section": _format_summary_section(state["summary"]),
        "history_section": _format_history_section(state["recent_messages"]),
        "context": state["context"],
        "question": state["question"],
    }

    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke(
        prompt_input,
        config={"callbacks": [handler]},
    )

    return {
        **state,
        "answer": answer,
    }


def save_memory(state: AgentState) -> AgentState:
    """Node 5: Save messages to PostgreSQL, trigger summarization if needed."""
    user_id = state["user_id"]
    session_id = state["session_id"]

    save_message(user_id, session_id, "human", state["question"])
    save_message(user_id, session_id, "assistant", state["answer"])

    summarized = False
    if should_summarize(user_id, session_id):
        summarize_and_compress(user_id, session_id)
        summarized = True

    trace_node(
        trace_id=state["trace_id"],
        node_name="save_memory",
        user_id=user_id,
        session_id=session_id,
        input_data={
            "question": state["question"],
            "cache_hit": state["cache_hit"],
        },
        output_data={
            "messages_saved": 2,
            "summarization_triggered": summarized,
        },
    )

    return state


def cache_response(state: AgentState) -> AgentState:
    """Node 6: Store answer in Redis if cache miss."""
    if not state["cache_hit"]:
        question_embedding = embed_query(state["question"])
        cache_store(
            question=state["question"],
            answer=state["answer"],
            sources=state["sources"],
            question_embedding=question_embedding,
        )

    trace_node(
        trace_id=state["trace_id"],
        node_name="cache_response",
        user_id=state["user_id"],
        session_id=state["session_id"],
        input_data={"cache_hit": state["cache_hit"]},
        output_data={"stored_in_cache": not state["cache_hit"]},
    )

    return state


def route_after_cache_check(state: AgentState) -> str:
    """Conditional edge — route based on cache hit/miss."""
    if state["cache_hit"]:
        return "save_memory"
    return "retrieve"