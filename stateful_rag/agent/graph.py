from langgraph.graph import StateGraph, END
from stateful_rag.agent.state import AgentState
from stateful_rag.agent.nodes import (
    load_memory,
    check_cache,
    retrieve,
    generate,
    save_memory,
    cache_response,
    route_after_cache_check,
)

import logging

# Suppress noisy LangGraph metadata warnings from Langfuse
logging.getLogger("langfuse").setLevel(logging.ERROR)


def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph agent.

    Graph structure:
        load_memory
            → check_cache
                → [cache hit]  → save_memory → cache_response → END
                → [cache miss] → retrieve → generate → save_memory → cache_response → END
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("load_memory", load_memory)
    graph.add_node("check_cache", check_cache)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("save_memory", save_memory)
    graph.add_node("cache_response", cache_response)

    # Entry point
    graph.set_entry_point("load_memory")

    # Edges
    graph.add_edge("load_memory", "check_cache")

    # Conditional routing after cache check
    graph.add_conditional_edges(
        "check_cache",
        route_after_cache_check,
        {
            "retrieve": "retrieve",
            "save_memory": "save_memory",
        }
    )

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "save_memory")
    graph.add_edge("save_memory", "cache_response")
    graph.add_edge("cache_response", END)

    return graph.compile()


# Singleton — build once, reuse
agent = build_graph()


def ask(
    question: str,
    user_id: str,
    session_id: str,
) -> dict:
    initial_state: AgentState = {
        "user_id": user_id,
        "session_id": session_id,
        "question": question,
        "trace_id": "",           # set by load_memory node
        "summary": "",
        "recent_messages": [],
        "context": "",
        "sources": [],
        "cache_hit": False,
        "answer": "",
    }

    final_state = agent.invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "sources": final_state["sources"],
        "cache_hit": final_state["cache_hit"],
        "user_id": user_id,
        "session_id": session_id,
    }