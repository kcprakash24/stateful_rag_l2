from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from stateful_rag.ingestion.loader import ParsedDocument


@dataclass
class DocumentChunk:
    """A single chunk ready for embedding."""
    chunk_id: str        # unique: filename_chunkindex
    text: str
    metadata: dict


def chunk_document(
    parsed_doc: ParsedDocument,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[DocumentChunk]:
    """
    Split a parsed document into overlapping chunks.

    Args:
        parsed_doc: Output from load_pdf()
        chunk_size: Max characters per chunk
        chunk_overlap: Characters shared between adjacent chunks

    Returns:
        List of DocumentChunk objects
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # These separators are tried in order — markdown-aware
        separators=[
            "\n## ",    # major section break
            "\n### ",   # subsection break
            "\n\n",     # paragraph break
            "\n",       # line break
            ". ",       # sentence break
            " ",        # word break
            "",         # character break (last resort)
        ],
        length_function=len,
    )

    raw_chunks = splitter.split_text(parsed_doc.markdown_text)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk = DocumentChunk(
            chunk_id=f"{parsed_doc.file_name}_chunk_{i:04d}",
            text=chunk_text.strip(),
            metadata={
                **parsed_doc.metadata,      # inherit parent metadata
                "chunk_index": i,
                "chunk_id": f"{parsed_doc.file_name}_chunk_{i:04d}",
                "total_chunks": len(raw_chunks),
            }
        )
        chunks.append(chunk)

    print(f"  '{parsed_doc.file_name}' → {len(chunks)} chunks "
          f"(avg {sum(len(c.text) for c in chunks) // len(chunks)} chars each)")

    return chunks


def chunk_documents(
    parsed_docs: list[ParsedDocument],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[DocumentChunk]:
    """Chunk multiple documents."""
    all_chunks = []
    for doc in parsed_docs:
        all_chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return all_chunks
