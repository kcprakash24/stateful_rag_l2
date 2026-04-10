from langchain_ollama import OllamaEmbeddings
from config import get_settings


def get_embeddings() -> OllamaEmbeddings:
    """
    Returns the embedding model.
    Always local via Ollama — never changes regardless of LLM provider setting.
    
    Returns:
        LangChain-compatible embedding object
    """
    settings = get_settings()

    return OllamaEmbeddings(
        model=settings.embedding_model,       # nomic-embed-text
        base_url=settings.ollama_base_url,    # http://localhost:11434
    )