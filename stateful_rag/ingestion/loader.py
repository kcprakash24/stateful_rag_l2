import pymupdf4llm
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ParsedDocument:
    """Clean container for a parsed PDF."""
    file_path: str
    file_name: str
    markdown_text: str
    num_pages: int
    metadata: dict


def load_pdf(file_path: str | Path) -> ParsedDocument:
    """
    Load a PDF and convert to clean markdown text.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        ParsedDocument with markdown content and metadata
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected PDF file, got: {path.suffix}")

    # Convert PDF to markdown
    # page_chunks=False means we get one big markdown string (we handle chunking ourselves)
    md_text = pymupdf4llm.to_markdown(
        str(path),
        page_chunks=False,      # single string output
        show_progress=False,
    )

    # Get page count separately
    import pymupdf
    doc = pymupdf.open(str(path))
    num_pages = len(doc)
    doc.close()

    return ParsedDocument(
        file_path=str(path.absolute()),
        file_name=path.name,
        markdown_text=md_text,
        num_pages=num_pages,
        metadata={
            "source": path.name,
            "file_path": str(path.absolute()),
            "num_pages": num_pages,
            "file_size_kb": round(path.stat().st_size / 1024, 2),
        }
    )


def load_pdfs_from_dir(dir_path: str | Path) -> list[ParsedDocument]:
    """Load all PDFs from a directory."""
    dir_path = Path(dir_path)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    pdf_files = list(dir_path.glob("*.pdf"))
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {dir_path}")
    
    print(f"Found {len(pdf_files)} PDF(s) in {dir_path}")
    
    return [load_pdf(pdf) for pdf in pdf_files]