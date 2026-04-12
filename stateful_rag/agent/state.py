# LangGraph models the pipeline as a directed graph where:

# Nodes = functions that do work and update state
# Edges = connections between nodes
# Conditional edges = routing logic based on state values
# State = a typed dict shared across all nodes

#                     ┌─────────────────┐
#                     │   load_memory   │
#                     └────────┬────────┘
#                              │
#                     ┌────────▼────────┐
#                     │  check_cache    │──── HIT ────────────────────┐
#                     └────────┬────────┘                             │
#                           MISS                                      │
#                     ┌────────▼────────┐                             │
#                     │    retrieve     │                             │
#                     └────────┬────────┘                             │
#                              │                                      │
#                     ┌────────▼────────┐                             │
#                     │    generate     │                             │
#                     └────────┬────────┘                             │
#                              │                                      │
#                     ┌────────▼────────┐                             │
#                     │  save_memory    │◄────────────────────────────┘
#                     └────────┬────────┘
#                              │
#                     ┌────────▼────────┐
#                     │  cache_response │
#                     └─────────────────┘


from typing import TypedDict

class AgentState(TypedDict):
    """
    Shared state passed between all LangGraph nodes.
    Every node reads from and writes to this object.
    """
    # Input
    user_id: str
    session_id: str
    question: str

    # Observability — shared across all nodes in one run
    trace_id: str

    # Memory
    summary: str                        # compressed older history
    recent_messages: list[dict]         # last N message pairs

    # Retrieval
    context: str                        # formatted retrieved chunks
    sources: list[dict]                 # chunk metadata for display

    # Cache
    cache_hit: bool                     # True if answer came from cache

    # Output
    answer: str